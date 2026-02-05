#ifndef UPSAMPLE_W8A32_H
#define UPSAMPLE_W8A32_H

#include <stdint.h>

void upsample_nearest2x_nchw_f32_w8a32(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    float* y);

#endif // UPSAMPLE_W8A32_H
