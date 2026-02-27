#ifndef WEIGHTS_LOADER_H
#define WEIGHTS_LOADER_H

#include <stdint.h>
#include <stddef.h>

#define MAX_TENSOR_DIMS 8

#define WEIGHTS_DTYPE_FLOAT32 0
#define WEIGHTS_DTYPE_INT8    1

typedef struct {
    char* name;              // 텐서 이름 (동적 할당)
    float* data;             // FP32 데이터 (dtype==0일 때만 사용)
    int8_t* data_int8;       // INT8 원시 데이터 (dtype==1일 때만 사용)
    float scale;             // INT8 디양자화: w_f32 = (float)w_int8 * scale
    unsigned char dtype;     // WEIGHTS_DTYPE_FLOAT32 or WEIGHTS_DTYPE_INT8
    int32_t ndim;
    int32_t shape[MAX_TENSOR_DIMS];
    size_t num_elements;
    unsigned char data_owned; // 1 = loader가 할당(해제 시 free), 0 = 외부(DDR) 참조
} tensor_info_t;

typedef struct {
    tensor_info_t* tensors;
    int32_t num_tensors;
} weights_loader_t;

int weights_init_from_memory(uintptr_t base_addr, size_t size, weights_loader_t* loader);

int weights_load_from_file(const char* bin_path, weights_loader_t* loader);

int weights_load_from_file_w8(const char* w8_path, weights_loader_t* loader);

#ifdef BARE_METAL
int weights_init_from_memory_w8(uintptr_t w8_base, size_t w8_size, weights_loader_t* loader);
#endif

const tensor_info_t* weights_find_tensor(const weights_loader_t* loader, const char* name);

const float* weights_get_tensor_data(weights_loader_t* loader, const char* name);

void* weights_get_tensor_for_conv(weights_loader_t* loader, const char* name, float* out_scale, int* out_is_int8);

void weights_free(weights_loader_t* loader);

#endif // WEIGHTS_LOADER_H
