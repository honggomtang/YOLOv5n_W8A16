#ifndef CONV2D_W8A16_H
#define CONV2D_W8A16_H

#include <stdint.h>

typedef struct {
    const void* ptr;
    float scale;   /* TODO: W8A16에서는 shift(int) 또는 scale 제거 */
    int is_int8;
} w8_conv_t_w8a16;

void conv2d_nchw_w8a16(
    const int16_t* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const int8_t* w, int32_t c_out, int32_t k_h, int32_t k_w,
    const int32_t* bias_or_null,
    uint32_t multiplier,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    int32_t groups,
    int16_t* y, int32_t h_out, int32_t w_out);

void conv2d_nchw_f32_w8a16(
    const float* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const float* w, int32_t c_out, int32_t k_h, int32_t k_w,
    const float* bias_or_null,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    int32_t groups,
    float* y, int32_t h_out, int32_t w_out);

void conv2d_nchw_f32_w8_w8a16(
    const float* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const int8_t* w, float scale, int32_t c_out, int32_t k_h, int32_t k_w,
    const float* bias_or_null,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    int32_t groups,
    float* y, int32_t h_out, int32_t w_out);

#endif // CONV2D_W8A16_H
