#ifndef SPPF_W8A32_H
#define SPPF_W8A32_H

#include <stdint.h>

void sppf_nchw_f32_w8a32(
    const float* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const void* cv1_w, float cv1_scale, int cv1_is_int8, int32_t cv1_c_out, const float* cv1_bias,
    const void* cv2_w, float cv2_scale, int cv2_is_int8, int32_t cv2_c_out, const float* cv2_bias,
    int32_t pool_k,
    float* y);

#endif // SPPF_W8A32_H
