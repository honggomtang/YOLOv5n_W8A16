#ifndef CONCAT_W8A32_H
#define CONCAT_W8A32_H

#include <stdint.h>

void concat_nchw_f32_w8a32(
    const float* x1, int32_t c1,
    const float* x2, int32_t c2,
    int32_t n, int32_t h, int32_t w,
    float* y);

void concat4_nchw_f32_w8a32(
    const float* x0, int32_t c0,
    const float* x1, int32_t c1,
    const float* x2, int32_t c2,
    const float* x3, int32_t c3,
    int32_t n, int32_t h, int32_t w,
    float* y);

#endif // CONCAT_W8A32_H
