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

/** W8A16: DDR/버퍼에서 24B 헤더만 파싱. 메타데이터만 채우고 img->data는 건드리지 않음 (호출자가 base+24를 int16*로 사용). */
int image_init_from_memory_a16(uintptr_t base_addr, size_t size, preprocessed_image_t* img);

// ===== 개발/테스트용: 파일 시스템에서 로드 =====
// 반환값: 0 성공, -1 실패
int image_load_from_bin(const char* bin_path, preprocessed_image_t* img);

/** W8A16: preprocessed_image_a16.bin 로드. 헤더는 img에, 전체 버퍼는 *out_buffer에 (호출자가 free). int16 데이터 = (char*)*out_buffer + 24 */
int image_load_from_bin_a16(const char* bin_path, preprocessed_image_t* img, void** out_buffer);

void image_free(preprocessed_image_t* img);

#endif // IMAGE_LOADER_H
