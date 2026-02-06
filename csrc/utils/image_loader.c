#include "image_loader.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifndef BARE_METAL
#include <stdio.h>
#endif

static inline void safe_read(void* dest, const uint8_t** src, size_t size) {
    memcpy(dest, *src, size);
    *src += size;
}

static int parse_image_data(const uint8_t* ptr, size_t data_len, preprocessed_image_t* img, int zero_copy) {
    const uint8_t* curr = ptr;
    const uint8_t* end = ptr + data_len;

    if (curr + 24 > end) return -1;
    // 헤더 24B
    uint32_t original_w, original_h, size;
    float scale;
    uint32_t pad_x, pad_y;
    
    safe_read(&original_w, &curr, 4);
    safe_read(&original_h, &curr, 4);
    safe_read(&scale, &curr, 4);
    safe_read(&pad_x, &curr, 4);
    safe_read(&pad_y, &curr, 4);
    safe_read(&size, &curr, 4);
    
    img->original_w = (int32_t)original_w;
    img->original_h = (int32_t)original_h;
    img->scale = scale;
    img->pad_x = (int32_t)pad_x;
    img->pad_y = (int32_t)pad_y;
    img->c = 3;
    img->h = (int32_t)size;
    img->w = (int32_t)size;
    
    size_t data_bytes = 3 * (size_t)size * (size_t)size * sizeof(float);
    if (curr + data_bytes > end) return -1;

    if (zero_copy) {
        img->data = (float*)curr;
        img->data_owned = 0;
    } else {
        img->data = (float*)malloc(data_bytes);
        if (!img->data) return -1;
        safe_read(img->data, &curr, data_bytes);
        img->data_owned = 1;
    }
    return 0;
}

int image_init_from_memory(uintptr_t base_addr, size_t size, preprocessed_image_t* img) {
    if (!img || size < 24) return -1;
    return parse_image_data((const uint8_t*)base_addr, size, img, 1);
}

/* W8A16: 24B 헤더만 파싱, 메타데이터만 채움. payload는 건드리지 않음 (zero-copy로 base+24를 int16*로 사용). */
int image_init_from_memory_a16(uintptr_t base_addr, size_t size, preprocessed_image_t* img) {
    const uint8_t* ptr = (const uint8_t*)base_addr;
    if (!img || size < 24u) return -1;
    /* 최소 크기: 24 + 3*640*640*2 */
    if (size < 24u + 3u * 640u * 640u * 2u) return -1;
    uint32_t original_w, original_h, sz;
    float scale;
    uint32_t pad_x, pad_y;
    memcpy(&original_w, ptr, 4); ptr += 4;
    memcpy(&original_h, ptr, 4); ptr += 4;
    memcpy(&scale, ptr, 4); ptr += 4;
    memcpy(&pad_x, ptr, 4); ptr += 4;
    memcpy(&pad_y, ptr, 4); ptr += 4;
    memcpy(&sz, ptr, 4);
    img->original_w = (int32_t)original_w;
    img->original_h = (int32_t)original_h;
    img->scale = scale;
    img->pad_x = (int32_t)pad_x;
    img->pad_y = (int32_t)pad_y;
    img->c = 3;
    img->h = (int32_t)sz;
    img->w = (int32_t)sz;
    img->data = NULL;
    img->data_owned = 0;
    return 0;
}

#ifndef BARE_METAL
int image_load_from_bin(const char* bin_path, preprocessed_image_t* img) {
    FILE* f = fopen(bin_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open image file: %s\n", bin_path);
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
    
    int ret = parse_image_data(buffer, file_size, img, 0);
    free(buffer);
    
    return ret;
}

/* W8A16: preprocessed_image_a16.bin (24B 헤더 + int16 데이터) 로드. *out_buffer에 전체 버퍼 반환(호출자 free). int16 데이터 = (int16_t*)((char*)*out_buffer + 24). */
int image_load_from_bin_a16(const char* bin_path, preprocessed_image_t* img, void** out_buffer) {
    FILE* f = fopen(bin_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open image file: %s\n", bin_path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_size < 24u + 3u * 640u * 640u * 2u) {
        fprintf(stderr, "Error: a16 file too small: %s\n", bin_path);
        fclose(f);
        return -1;
    }
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
    /* 헤더만 파싱 (parse_image_data는 float 크기 기대하므로 수동 파싱) */
    const uint8_t* curr = buffer;
    uint32_t original_w, original_h, sz;
    float scale;
    uint32_t pad_x, pad_y;
    memcpy(&original_w, curr, 4); curr += 4;
    memcpy(&original_h, curr, 4); curr += 4;
    memcpy(&scale, curr, 4); curr += 4;
    memcpy(&pad_x, curr, 4); curr += 4;
    memcpy(&pad_y, curr, 4); curr += 4;
    memcpy(&sz, curr, 4);
    img->original_w = (int32_t)original_w;
    img->original_h = (int32_t)original_h;
    img->scale = scale;
    img->pad_x = (int32_t)pad_x;
    img->pad_y = (int32_t)pad_y;
    img->c = 3;
    img->h = (int32_t)sz;
    img->w = (int32_t)sz;
    img->data = NULL;
    img->data_owned = 0;
    *out_buffer = buffer;
    return 0;
}
#endif

void image_free(preprocessed_image_t* img) {
    if (!img) return;
    if (img->data_owned && img->data) {
        free(img->data);
        img->data = NULL;
    }
}
