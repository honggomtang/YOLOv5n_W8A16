#ifndef MAXPOOL2D_W8A32_H
#define MAXPOOL2D_W8A32_H

#include <stdint.h>

void maxpool2d_nchw_f32_w8a32(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    int32_t k, int32_t stride, int32_t pad,
    float* y, int32_t out_h, int32_t out_w);

#endif // MAXPOOL2D_W8A32_H
