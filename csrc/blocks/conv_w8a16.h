#ifndef CONV_W8A16_H
#define CONV_W8A16_H

#include <stdint.h>

void conv_block_nchw_f32_w8a16(
    const float* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const void* w, float w_scale, int w_is_int8,
    int32_t c_out, int32_t k_h, int32_t k_w,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    const float* bias,
    float* y, int32_t h_out, int32_t w_out);

void conv_block_nchw_w8a16(
    const int16_t* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const int8_t* w, int32_t c_out, int32_t k_h, int32_t k_w,
    const int32_t* bias_or_null, uint32_t multiplier,
    int32_t stride_h, int32_t stride_w, int32_t pad_h, int32_t pad_w,
    int16_t* y, int32_t h_out, int32_t w_out);

#endif // CONV_W8A16_H
