#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    float* data;         // 이미지 데이터 (C, H, W) - NCHW 형식
    int32_t c, h, w;     // 채널, 높이, 너비
    int32_t original_w, original_h;  // 원본 이미지 크기
    float scale;         // 리사이즈 스케일
    int32_t pad_x, pad_y;  // 패딩 위치
    unsigned char data_owned; // 1 = loader가 할당(해제 시 free), 0 = 외부(DDR) 참조
} preprocessed_image_t;

int image_init_from_memory(uintptr_t base_addr, size_t size, preprocessed_image_t* img);

int image_init_from_memory_a16(uintptr_t base_addr, size_t size, preprocessed_image_t* img);
int image_load_from_bin(const char* bin_path, preprocessed_image_t* img);
int image_load_from_bin_a16(const char* bin_path, preprocessed_image_t* img, void** out_buffer);

void image_free(preprocessed_image_t* img);

#endif // IMAGE_LOADER_H
