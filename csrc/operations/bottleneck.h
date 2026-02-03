#ifndef BOTTLENECK_H
#define BOTTLENECK_H

#include <stdint.h>

/* W8A32: cv1_w/cv2_w는 void* (float* 또는 int8_t*), scale/is_int8로 구분 */
void bottleneck_nchw_f32(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    const void* cv1_w, float cv1_scale, int cv1_is_int8, int32_t cv1_c_out, const float* cv1_bias,
    const void* cv2_w, float cv2_scale, int cv2_is_int8, int32_t cv2_c_out, const float* cv2_bias,
    int32_t shortcut,  // 1=add residual, 0=no shortcut
    float* y);

#endif // BOTTLENECK_H
