#ifndef SILU_W8A16_H
#define SILU_W8A16_H

#include <stdint.h>

void silu_nchw_f32_w8a16(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    float* y);

/* W8A16 전용: int16_t (Q6.10) 입출력, 65536 LUT 기반 SiLU */
void silu_nchw_w8a16(
    const int16_t* x, int32_t n, int32_t c, int32_t h, int32_t w,
    int16_t* y);

#endif // SILU_W8A16_H
