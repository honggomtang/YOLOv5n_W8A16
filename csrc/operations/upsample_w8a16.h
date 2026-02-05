#ifndef UPSAMPLE_W8A16_H
#define UPSAMPLE_W8A16_H

#include <stdint.h>

void upsample_nearest2x_nchw_f32_w8a16(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    float* y);

/* W8A16 전용: 입출력 모두 int16_t (Q6.10). 데이터 타입만 int16, nearest 2x. */
void upsample_nearest2x_nchw_w8a16(
    const int16_t* x, int32_t n, int32_t c, int32_t h, int32_t w,
    int16_t* y);

#endif // UPSAMPLE_W8A16_H
