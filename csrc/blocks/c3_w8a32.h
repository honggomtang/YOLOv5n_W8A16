#ifndef C3_W8A32_H
#define C3_W8A32_H

#include <stdint.h>

void c3_nchw_f32_w8a32(
    const float* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const void* cv1_w, float cv1_scale, int cv1_is_int8, int32_t cv1_c_out, const float* cv1_bias,
    const void* cv2_w, float cv2_scale, int cv2_is_int8, int32_t cv2_c_out, const float* cv2_bias,
    const void* cv3_w, float cv3_scale, int cv3_is_int8, int32_t cv3_c_out, const float* cv3_bias,
    int32_t n_bottleneck,
    const void** bn_cv1_w, const float* bn_cv1_scale, const int* bn_cv1_is_int8,
    const float* const* bn_cv1_bias,
    const void** bn_cv2_w, const float* bn_cv2_scale, const int* bn_cv2_is_int8,
    const float* const* bn_cv2_bias,
    int32_t shortcut,
    float* y);

#endif // C3_W8A32_H
