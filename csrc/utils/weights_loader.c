#include "weights_loader.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#ifndef WEIGHTS_WARN_MISSING
#define WEIGHTS_WARN_MISSING 1
#endif

static inline void safe_read(void* dest, const uint8_t** src, size_t size) {
    memcpy(dest, *src, size);
    *src += size;
}

// RISC-V 등: 비정렬 주소에서 4바이트 읽기 (바이트 단위로만 접근 → trap 방지)
static inline uint32_t read_u32_unaligned(const uint8_t** src) {
    const uint8_t* p = *src;
    uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    *src = p + 4;
    return v;
}

static int parse_weights_data(const uint8_t* ptr, size_t data_len, weights_loader_t* loader, int zero_copy) {
    const uint8_t* curr = ptr;
    const uint8_t* end = ptr + data_len;

    if (curr + 4 > end) return -1;
    uint32_t num_tensors;
    safe_read(&num_tensors, &curr, 4);

    loader->num_tensors = (int32_t)num_tensors;
    loader->tensors = (tensor_info_t*)calloc(num_tensors, sizeof(tensor_info_t));
    if (!loader->tensors) return -1;
    loader->dequant_pool_base = NULL;
    loader->dequant_buf_cap = 0;
    loader->dequant_pool_next = 0;

    for (int i = 0; i < (int)num_tensors; i++) {
        tensor_info_t* t = &loader->tensors[i];
        t->dtype = WEIGHTS_DTYPE_FLOAT32;
        t->data_int8 = NULL;
        t->scale = 0.f;

        if (curr + 4 > end) return -1;
        uint32_t key_len;
        safe_read(&key_len, &curr, 4);
        if (key_len > 1024) return -1;
        if (curr + key_len > end) return -1;

        t->name = (char*)malloc(key_len + 1);
        if (!t->name) return -1;
        safe_read(t->name, &curr, key_len);
        t->name[key_len] = '\0';

        if (curr + 4 > end) return -1;
        uint32_t ndim = read_u32_unaligned(&curr);
        t->ndim = (int32_t)ndim;

        if (ndim > MAX_TENSOR_DIMS) return -1;

        if (curr + ndim * 4 > end) return -1;
        t->num_elements = 1;
        for (int j = 0; j < (int)ndim; j++) {
            uint32_t dim_val = read_u32_unaligned(&curr);
            t->shape[j] = (int32_t)dim_val;
            t->num_elements *= dim_val;
        }

        {
            uintptr_t u = (uintptr_t)curr;
            u = (u + 3u) & ~(uintptr_t)3u;
            curr = (const uint8_t*)u;
        }

        size_t data_bytes = t->num_elements * sizeof(float);
        if (curr + data_bytes > end) return -1;

        if (zero_copy) {
            t->data = (float*)curr;
            if ((uintptr_t)t->data % 4 != 0)
                return -1;
            curr = (const uint8_t*)((const float*)curr + t->num_elements);
            t->data_owned = 0;
        } else {
            t->data = (float*)malloc(data_bytes);
            if (!t->data) return -1;
            safe_read(t->data, &curr, data_bytes);
            t->data_owned = 1;
        }
    }

    return 0;
}

/* W8A32: weights_w8.bin 파싱. scale은 INT8 텐서 헤더에 포함 (A: 텐서별 매칭, 순서 독립). */
static int parse_weights_w8(const uint8_t* w8_ptr, size_t w8_len,
                            weights_loader_t* loader, int zero_copy) {
    const uint8_t* curr = w8_ptr;
    const uint8_t* end = w8_ptr + w8_len;
    size_t max_int8_elems = 0;

    if (curr + 4 > end) return -1;
    uint32_t num_tensors;
    safe_read(&num_tensors, &curr, 4);

    loader->num_tensors = (int32_t)num_tensors;
    loader->tensors = (tensor_info_t*)calloc(num_tensors, sizeof(tensor_info_t));
    if (!loader->tensors) return -1;
    loader->dequant_pool_base = NULL;
    loader->dequant_buf_cap = 0;
    loader->dequant_pool_next = 0;

    for (int i = 0; i < (int)num_tensors; i++) {
        tensor_info_t* t = &loader->tensors[i];
        t->data = NULL;
        t->data_int8 = NULL;
        t->scale = 0.f;

        if (curr + 4 > end) return -1;
        uint32_t key_len;
        safe_read(&key_len, &curr, 4);
        if (key_len > 1024) return -1;
        if (curr + key_len > end) return -1;

        t->name = (char*)malloc(key_len + 1);
        if (!t->name) return -1;
        safe_read(t->name, &curr, key_len);
        t->name[key_len] = '\0';

        if (curr + 4 > end) return -1;
        uint32_t ndim = read_u32_unaligned(&curr);
        t->ndim = (int32_t)ndim;
        if (ndim > MAX_TENSOR_DIMS) return -1;

        if (curr + ndim * 4 > end) return -1;
        t->num_elements = 1;
        for (int j = 0; j < (int)ndim; j++) {
            uint32_t dim_val = read_u32_unaligned(&curr);
            t->shape[j] = (int32_t)dim_val;
            t->num_elements *= dim_val;
        }

        if (curr + 1 > end) return -1;
        t->dtype = (unsigned char)curr[0];
        curr += 1;

        if (t->dtype == WEIGHTS_DTYPE_FLOAT32) {
            uintptr_t u = (uintptr_t)curr;
            u = (u + 3u) & ~(uintptr_t)3u;
            curr = (const uint8_t*)u;
            size_t data_bytes = t->num_elements * sizeof(float);
            if (curr + data_bytes > end) return -1;
            if (zero_copy) {
                t->data = (float*)curr;
                curr += data_bytes;
                t->data_owned = 0;
            } else {
                t->data = (float*)malloc(data_bytes);
                if (!t->data) return -1;
                safe_read(t->data, &curr, data_bytes);
                t->data_owned = 1;
            }
        } else if (t->dtype == WEIGHTS_DTYPE_INT8) {
            if (curr + 4 > end) return -1;
            safe_read(&t->scale, &curr, 4);  /* scale in w8 (D: 4B 정렬 유지) */
            {
                uintptr_t u = (uintptr_t)curr;
                u = (u + 3u) & ~(uintptr_t)3u;
                curr = (const uint8_t*)u;
            }
            size_t data_bytes = t->num_elements * (size_t)1;
            if (curr + data_bytes > end) return -1;
            if (t->num_elements > max_int8_elems)
                max_int8_elems = t->num_elements;
            if (zero_copy) {
                t->data_int8 = (int8_t*)curr;
                curr += data_bytes;
                t->data_owned = 0;
            } else {
                t->data_int8 = (int8_t*)malloc(data_bytes);
                if (!t->data_int8) return -1;
                safe_read(t->data_int8, &curr, data_bytes);
                t->data_owned = 1;
            }
        } else
            return -1;
    }

    /* B: 버퍼 풀 — c3/detect 등에서 여러 가중치를 동시에 쓰므로 슬롯 여러 개 필요 */
    if (max_int8_elems > 0) {
        size_t pool_bytes = (size_t)WEIGHTS_DEQUANT_POOL_SIZE * max_int8_elems * sizeof(float);
        loader->dequant_pool_base = (float*)malloc(pool_bytes);
        if (!loader->dequant_pool_base) return -1;
        loader->dequant_buf_cap = max_int8_elems;
    }
    return 0;
}

int weights_init_from_memory(uintptr_t base_addr, size_t size, weights_loader_t* loader) {
    if (size == 0) return -1;
    return parse_weights_data((const uint8_t*)base_addr, size, loader, 1);
}

#ifdef BARE_METAL
int weights_init_from_memory_w8(uintptr_t w8_base, size_t w8_size, weights_loader_t* loader) {
    if (w8_size == 0) return -1;
    return parse_weights_w8((const uint8_t*)w8_base, w8_size, loader, 1);
}
#endif

#ifndef BARE_METAL
int weights_load_from_file(const char* bin_path, weights_loader_t* loader) {
    FILE* f = fopen(bin_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", bin_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t* buffer = (uint8_t*)malloc(file_size);
    if (!buffer) {
        fclose(f);
        return -1;
    }

    if (fread(buffer, 1, file_size, f) != file_size) {
        free(buffer);
        fclose(f);
        return -1;
    }
    fclose(f);

    int ret = parse_weights_data(buffer, file_size, loader, 0);
    free(buffer);
    
    if (ret != 0) {
        weights_free(loader);
    }
    return ret;
}

int weights_load_from_file_w8(const char* w8_path, weights_loader_t* loader) {
    FILE* fw = fopen(w8_path, "rb");
    if (!fw) {
        fprintf(stderr, "Error: Cannot open %s\n", w8_path);
        return -1;
    }
    fseek(fw, 0, SEEK_END);
    long w8_size = ftell(fw);
    fseek(fw, 0, SEEK_SET);
    if (w8_size <= 0) { fclose(fw); return -1; }
    uint8_t* w8_buf = (uint8_t*)malloc((size_t)w8_size);
    if (!w8_buf) { fclose(fw); return -1; }
    if (fread(w8_buf, 1, (size_t)w8_size, fw) != (size_t)w8_size) {
        free(w8_buf); fclose(fw); return -1;
    }
    fclose(fw);

    int ret = parse_weights_w8(w8_buf, (size_t)w8_size, loader, 0);
    free(w8_buf);
    if (ret != 0) {
        weights_free(loader);
        return ret;
    }
    return 0;
}
#endif

const tensor_info_t* weights_find_tensor(const weights_loader_t* loader, const char* name) {
    char search_name[512];
    
    for (int i = 0; i < loader->num_tensors; i++) {
        if (strcmp(loader->tensors[i].name, name) == 0) {
            return &loader->tensors[i];
        }
    }

    if (strncmp(name, "model.", 6) == 0) {
        snprintf(search_name, sizeof(search_name), "model.model.%s", name);
        for (int i = 0; i < loader->num_tensors; i++) {
            if (strcmp(loader->tensors[i].name, search_name) == 0) {
                return &loader->tensors[i];
            }
        }
    }

    return NULL;
}

const float* weights_get_tensor_data(weights_loader_t* loader, const char* name) {
    const tensor_info_t* t = weights_find_tensor(loader, name);
    if (!t) {
#if WEIGHTS_WARN_MISSING && !defined(BARE_METAL)
        fprintf(stderr, "Warning: Weight not found: %s\n", name);
#endif
        return NULL;
    }
    if (t->dtype == WEIGHTS_DTYPE_INT8 && t->data_int8 && loader->dequant_pool_base && t->num_elements <= loader->dequant_buf_cap) {
        /* 슬롯 하나 사용 (round-robin) — c3/detect에서 여러 W() 호출이 인자 평가 시 순차 실행되므로 서로 다른 슬롯에 채워짐 */
        int slot = loader->dequant_pool_next;
        loader->dequant_pool_next = (slot + 1) % WEIGHTS_DEQUANT_POOL_SIZE;
        float* dst = loader->dequant_pool_base + (size_t)slot * loader->dequant_buf_cap;
        const int8_t* src = t->data_int8;
        float s = t->scale;
        size_t n = t->num_elements;
        for (size_t i = 0; i < n; i++)
            dst[i] = (float)src[i] * s;
        return dst;
    }
    return t->data;
}

void* weights_get_tensor_for_conv(weights_loader_t* loader, const char* name, float* out_scale, int* out_is_int8) {
    const tensor_info_t* t = weights_find_tensor(loader, name);
    if (!t) {
#if WEIGHTS_WARN_MISSING && !defined(BARE_METAL)
        fprintf(stderr, "Warning: Weight not found: %s\n", name);
#endif
        if (out_scale) *out_scale = 0.f;
        if (out_is_int8) *out_is_int8 = 0;
        return NULL;
    }
    if (t->dtype == WEIGHTS_DTYPE_INT8 && t->data_int8) {
        if (out_scale) *out_scale = t->scale;
        if (out_is_int8) *out_is_int8 = 1;
        return (void*)t->data_int8;
    }
    if (out_scale) *out_scale = 0.f;
    if (out_is_int8) *out_is_int8 = 0;
    return (void*)t->data;
}

void weights_free(weights_loader_t* loader) {
    if (!loader || !loader->tensors) return;

    for (int i = 0; i < loader->num_tensors; i++) {
        tensor_info_t* t = &loader->tensors[i];
        if (t->name) free(t->name);
        if (t->data_owned) {
            if (t->dtype == WEIGHTS_DTYPE_INT8 && t->data_int8)
                free(t->data_int8);
            else if (t->data)
                free(t->data);
        }
    }
    if (loader->dequant_pool_base) free(loader->dequant_pool_base);
    free(loader->tensors);
    loader->tensors = NULL;
    loader->num_tensors = 0;
    loader->dequant_pool_base = NULL;
    loader->dequant_buf_cap = 0;
    loader->dequant_pool_next = 0;
}
