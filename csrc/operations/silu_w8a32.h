#ifndef SILU_W8A32_H
#define SILU_W8A32_H

#include <stdint.h>

void silu_nchw_f32_w8a32(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    float* y);

#endif // SILU_W8A32_H
