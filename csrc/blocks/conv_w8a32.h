#ifndef CONV_W8A32_H
#define CONV_W8A32_H

#include <stdint.h>

void conv_block_nchw_f32_w8a32(
    const float* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const void* w, float w_scale, int w_is_int8,
    int32_t c_out, int32_t k_h, int32_t k_w,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    const float* bias,
    float* y, int32_t h_out, int32_t w_out);

#endif // CONV_W8A32_H
