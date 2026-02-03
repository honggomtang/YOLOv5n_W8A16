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

/* C3 등에서 동시에 쓰는 가중치 최대 개수 (cv1,cv2,cv3 + n×cv1w + n×cv2w, n=3 → 9) */
#define WEIGHTS_DEQUANT_POOL_SIZE 10

// 가중치 로더 구조체
typedef struct {
    tensor_info_t* tensors;
    int32_t num_tensors;
    float* dequant_pool_base;  /* INT8 → FP32 풀: pool_base[slot * dequant_buf_cap] */
    size_t dequant_buf_cap;    /* 슬롯당 원소 개수 (max INT8 텐서 크기) */
    int dequant_pool_next;     /* 다음에 쓸 슬롯 (round-robin) */
} weights_loader_t;

int weights_init_from_memory(uintptr_t base_addr, size_t size, weights_loader_t* loader);

int weights_load_from_file(const char* bin_path, weights_loader_t* loader);

/* W8A32: weights_w8.bin 로드 (scale은 w8 내부 텐서 헤더에 포함). INT8 텐서는 get 시 디양자화해 float* 반환. */
int weights_load_from_file_w8(const char* w8_path, weights_loader_t* loader);

#ifdef BARE_METAL
int weights_init_from_memory_w8(uintptr_t w8_base, size_t w8_size, weights_loader_t* loader);
#endif

// 특정 이름의 텐서 찾기
// 반환값: 텐서 포인터, 없으면 NULL
const tensor_info_t* weights_find_tensor(const weights_loader_t* loader, const char* name);

/* INT8 시 풀 슬롯을 채우므로 loader는 non-const */
const float* weights_get_tensor_data(weights_loader_t* loader, const char* name);

/* W8A32 즉시 복원용: (ptr, scale, is_int8) 반환. conv_block/c3/detect에서 사용. */
void* weights_get_tensor_for_conv(weights_loader_t* loader, const char* name, float* out_scale, int* out_is_int8);

void weights_free(weights_loader_t* loader);

#endif // WEIGHTS_LOADER_H
