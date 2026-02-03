#ifndef DETECT_H
#define DETECT_H

#include <stdint.h>

/* W8A32: m0_w/m1_w/m2_w는 void*, scale/is_int8로 구분 */
void detect_nchw_f32(
    const float* p3, int32_t p3_c, int32_t p3_h, int32_t p3_w,
    const float* p4, int32_t p4_c, int32_t p4_h, int32_t p4_w,
    const float* p5, int32_t p5_c, int32_t p5_h, int32_t p5_w,
    const void* m0_w, float m0_scale, int m0_is_int8, const float* m0_b,
    const void* m1_w, float m1_scale, int m1_is_int8, const float* m1_b,
    const void* m2_w, float m2_scale, int m2_is_int8, const float* m2_b,
    float* p3_out, float* p4_out, float* p5_out);

#endif /* DETECT_H */
