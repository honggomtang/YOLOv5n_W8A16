#include "c3_w8a32.h"
#include "../operations/conv2d_w8a32.h"
#include "../operations/silu_w8a32.h"
#include "../operations/bottleneck_w8a32.h"
#include "../operations/concat_w8a32.h"
#include "../utils/feature_pool.h"
#include "../utils/timing.h"
#include <stdint.h>
#ifdef BARE_METAL
#include "xil_printf.h"
#endif

static void conv1x1_w8a32(
    const float* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const void* w_ptr, float w_scale, int w_is_int8, int32_t c_out, const float* bias,
    float* y)
{
    if (w_is_int8) {
        conv2d_nchw_f32_w8_w8a32(x, n, c_in, h, w,
                           (const int8_t*)w_ptr, w_scale, c_out, 1, 1,
                           bias, 1, 1, 0, 0, 1,
                           y, h, w);
    } else {
        conv2d_nchw_f32_w8a32(x, n, c_in, h, w,
                        (const float*)w_ptr, c_out, 1, 1,
                        bias, 1, 1, 0, 0, 1,
                        y, h, w);
    }
    silu_nchw_f32_w8a32(y, n, c_out, h, w, y);
}

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
    float* y)
{
    size_t cv1_bytes = (size_t)n * (size_t)cv1_c_out * (size_t)h * (size_t)w * sizeof(float);
    size_t cv2_bytes = (size_t)n * (size_t)cv2_c_out * (size_t)h * (size_t)w * sizeof(float);
    size_t cat_bytes = (size_t)n * (size_t)(cv1_c_out + cv2_c_out) * (size_t)h * (size_t)w * sizeof(float);

    float* concat_out = (float*)feature_pool_alloc(cat_bytes);
    float* cv1_out = (float*)feature_pool_alloc(cv1_bytes);
    float* cv2_out = (float*)feature_pool_alloc(cv2_bytes);
    float* bn_a = (float*)feature_pool_alloc(cv1_bytes);
    float* bn_b = (float*)feature_pool_alloc(cv1_bytes);

    if (!concat_out || !cv1_out || !cv2_out || !bn_a || !bn_b) {
#ifdef BARE_METAL
        xil_printf("C3 pool alloc failed cat=%08X cv1=%08X cv2=%08X bn_a=%08X bn_b=%08X\n",
                   (unsigned)(uintptr_t)concat_out, (unsigned)(uintptr_t)cv1_out,
                   (unsigned)(uintptr_t)cv2_out, (unsigned)(uintptr_t)bn_a,
                   (unsigned)(uintptr_t)bn_b);
#endif
        if (bn_b) feature_pool_free(bn_b);
        if (bn_a) feature_pool_free(bn_a);
        if (cv2_out) feature_pool_free(cv2_out);
        if (cv1_out) feature_pool_free(cv1_out);
        if (concat_out) feature_pool_free(concat_out);
        return;
    }

    yolo_timing_begin("cv1");
    conv1x1_w8a32(x, n, c_in, h, w, cv1_w, cv1_scale, cv1_is_int8, cv1_c_out, cv1_bias, cv1_out);
    yolo_timing_end();
    yolo_timing_begin("cv2");
    conv1x1_w8a32(x, n, c_in, h, w, cv2_w, cv2_scale, cv2_is_int8, cv2_c_out, cv2_bias, cv2_out);
    yolo_timing_end();
    yolo_timing_begin("bottleneck");
    const float* bn_in = cv1_out;
    float* bn_out = bn_a;
    for (int32_t i = 0; i < n_bottleneck; i++) {
        bn_out = (i % 2 == 0) ? bn_a : bn_b;
        bottleneck_nchw_f32_w8a32(
            bn_in, n, cv1_c_out, h, w,
            bn_cv1_w[i], bn_cv1_scale[i], bn_cv1_is_int8[i], cv1_c_out, bn_cv1_bias[i],
            bn_cv2_w[i], bn_cv2_scale[i], bn_cv2_is_int8[i], cv1_c_out, bn_cv2_bias[i],
            shortcut,
            bn_out);
        bn_in = bn_out;
    }
    yolo_timing_end();
    yolo_timing_begin("concat");
    concat_nchw_f32_w8a32(bn_out, cv1_c_out, cv2_out, cv2_c_out, n, h, w, concat_out);
    yolo_timing_end();
    yolo_timing_begin("cv3");
    conv1x1_w8a32(concat_out, n, cv1_c_out + cv2_c_out, h, w, cv3_w, cv3_scale, cv3_is_int8, cv3_c_out, cv3_bias, y);
    yolo_timing_end();

    feature_pool_free(concat_out);
    feature_pool_free(bn_b);
    feature_pool_free(bn_a);
    feature_pool_free(cv2_out);
    feature_pool_free(cv1_out);
}
