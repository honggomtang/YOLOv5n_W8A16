#ifndef CONV_H
#define CONV_H

#include <stdint.h>

/* w: float* 또는 int8_t* (w_is_int8에 따름). w_scale: INT8일 때만 사용. */
void conv_block_nchw_f32(
    const float* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const void* w, float w_scale, int w_is_int8,
    int32_t c_out, int32_t k_h, int32_t k_w,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    const float* bias,
    float* y, int32_t h_out, int32_t w_out);

#endif // CONV_H
