#include "detect_w8a32.h"
#include "../operations/conv2d_w8a32.h"
#include "../utils/timing.h"

void detect_nchw_f32_w8a32(
    const float* p3, int32_t p3_c, int32_t p3_h, int32_t p3_w,
    const float* p4, int32_t p4_c, int32_t p4_h, int32_t p4_w,
    const float* p5, int32_t p5_c, int32_t p5_h, int32_t p5_w,
    const void* m0_w, float m0_scale, int m0_is_int8, const float* m0_b,
    const void* m1_w, float m1_scale, int m1_is_int8, const float* m1_b,
    const void* m2_w, float m2_scale, int m2_is_int8, const float* m2_b,
    float* p3_out, float* p4_out, float* p5_out)
{
    yolo_timing_begin("detect");
    if (m0_is_int8) {
        conv2d_nchw_f32_w8_w8a32(p3, 1, p3_c, p3_h, p3_w,
            (const int8_t*)m0_w, m0_scale, 255, 1, 1, m0_b, 1, 1, 0, 0, 1,
            p3_out, p3_h, p3_w);
    } else {
        conv2d_nchw_f32_w8a32(p3, 1, p3_c, p3_h, p3_w,
            (const float*)m0_w, 255, 1, 1, m0_b, 1, 1, 0, 0, 1,
            p3_out, p3_h, p3_w);
    }
    if (m1_is_int8) {
        conv2d_nchw_f32_w8_w8a32(p4, 1, p4_c, p4_h, p4_w,
            (const int8_t*)m1_w, m1_scale, 255, 1, 1, m1_b, 1, 1, 0, 0, 1,
            p4_out, p4_h, p4_w);
    } else {
        conv2d_nchw_f32_w8a32(p4, 1, p4_c, p4_h, p4_w,
            (const float*)m1_w, 255, 1, 1, m1_b, 1, 1, 0, 0, 1,
            p4_out, p4_h, p4_w);
    }
    if (m2_is_int8) {
        conv2d_nchw_f32_w8_w8a32(p5, 1, p5_c, p5_h, p5_w,
            (const int8_t*)m2_w, m2_scale, 255, 1, 1, m2_b, 1, 1, 0, 0, 1,
            p5_out, p5_h, p5_w);
    } else {
        conv2d_nchw_f32_w8a32(p5, 1, p5_c, p5_h, p5_w,
            (const float*)m2_w, 255, 1, 1, m2_b, 1, 1, 0, 0, 1,
            p5_out, p5_h, p5_w);
    }
    yolo_timing_end();
}
