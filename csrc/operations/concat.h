#ifndef CONCAT_H
#define CONCAT_H

#include <stdint.h>

// Channel 차원 기준 concatenation 연산
void concat_nchw_f32(
    const float* x1, int32_t c1,
    const float* x2, int32_t c2,
    int32_t n, int32_t h, int32_t w,
    float* y);


// 4개 텐서 concat (NCHW, channel)
void concat4_nchw_f32(
    const float* x0, int32_t c0,
    const float* x1, int32_t c1,
    const float* x2, int32_t c2,
    const float* x3, int32_t c3,
    int32_t n, int32_t h, int32_t w,
    float* y);

#endif // CONCAT_H
