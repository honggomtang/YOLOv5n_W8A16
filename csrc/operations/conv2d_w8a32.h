#ifndef CONV2D_W8A32_H
#define CONV2D_W8A32_H

#include <stdint.h>

/* W8A32: conv에 넘길 가중치 (float* 또는 int8_t* + scale) */
typedef struct {
    const void* ptr;
    float scale;
    int is_int8;
} w8_conv_t_w8a32;

void conv2d_nchw_f32_w8a32(
    const float* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const float* w, int32_t c_out, int32_t k_h, int32_t k_w,
    const float* bias_or_null,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    int32_t groups,
    float* y, int32_t h_out, int32_t w_out);

/* W8A32: 가중치 INT8, 루프 내 (float)w_int8*scale 로 즉시 복원 */
void conv2d_nchw_f32_w8_w8a32(
    const float* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const int8_t* w, float scale, int32_t c_out, int32_t k_h, int32_t k_w,
    const float* bias_or_null,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    int32_t groups,
    float* y, int32_t h_out, int32_t w_out);

#endif // CONV2D_W8A32_H
