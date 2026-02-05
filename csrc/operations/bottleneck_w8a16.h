#ifndef BOTTLENECK_W8A16_H
#define BOTTLENECK_W8A16_H

#include <stdint.h>

void bottleneck_nchw_f32_w8a16(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    const void* cv1_w, float cv1_scale, int cv1_is_int8, int32_t cv1_c_out, const float* cv1_bias,
    const void* cv2_w, float cv2_scale, int cv2_is_int8, int32_t cv2_c_out, const float* cv2_bias,
    int32_t shortcut,
    float* y);

/* W8A16 전용: 입출력 int16_t (Q6.10). multiplier = Scale_W*65536. */
void bottleneck_nchw_w8a16(
    const int16_t* x, int32_t n, int32_t c, int32_t h, int32_t w,
    const int8_t* cv1_w, int32_t cv1_c_out, const int32_t* cv1_bias, uint32_t cv1_mult,
    const int8_t* cv2_w, int32_t cv2_c_out, const int32_t* cv2_bias, uint32_t cv2_mult,
    int32_t shortcut,
    int16_t* y);

#endif // BOTTLENECK_W8A16_H
