#include "sppf_w8a32.h"
#include "../operations/conv2d_w8a32.h"
#include "../operations/silu_w8a32.h"
#include "../operations/maxpool2d_w8a32.h"
#include "../operations/concat_w8a32.h"
#include "../utils/feature_pool.h"
#include "../utils/timing.h"

void sppf_nchw_f32_w8a32(
    const float* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const void* cv1_w, float cv1_scale, int cv1_is_int8, int32_t cv1_c_out, const float* cv1_bias,
    const void* cv2_w, float cv2_scale, int cv2_is_int8, int32_t cv2_c_out, const float* cv2_bias,
    int32_t pool_k,
    float* y)
{
    const int32_t pad = pool_k / 2;

    size_t x1_bytes = (size_t)n * (size_t)cv1_c_out * (size_t)h * (size_t)w * sizeof(float);
    size_t cat_bytes = (size_t)n * (size_t)(4 * cv1_c_out) * (size_t)h * (size_t)w * sizeof(float);

    float* x1 = (float*)feature_pool_alloc(x1_bytes);
    float* y1 = (float*)feature_pool_alloc(x1_bytes);
    float* y2 = (float*)feature_pool_alloc(x1_bytes);
    float* y3 = (float*)feature_pool_alloc(x1_bytes);
    float* cat = (float*)feature_pool_alloc(cat_bytes);

    if (!x1 || !y1 || !y2 || !y3 || !cat) {
        if (cat) feature_pool_free(cat);
        if (y3) feature_pool_free(y3);
        if (y2) feature_pool_free(y2);
        if (y1) feature_pool_free(y1);
        if (x1) feature_pool_free(x1);
        return;
    }
    yolo_timing_begin("cv1");
    if (cv1_is_int8 && cv1_w) {
        conv2d_nchw_f32_w8_w8a32(x, n, c_in, h, w,
                           (const int8_t*)cv1_w, cv1_scale, cv1_c_out, 1, 1,
                           cv1_bias, 1, 1, 0, 0, 1,
                           x1, h, w);
    } else if (cv1_w) {
        conv2d_nchw_f32_w8a32(x, n, c_in, h, w,
                        (const float*)cv1_w, cv1_c_out, 1, 1,
                        cv1_bias, 1, 1, 0, 0, 1,
                        x1, h, w);
    }
    silu_nchw_f32_w8a32(x1, n, cv1_c_out, h, w, x1);
    yolo_timing_end();

    yolo_timing_begin("maxpool");
    maxpool2d_nchw_f32_w8a32(x1, n, cv1_c_out, h, w, pool_k, 1, pad, y1, h, w);
    maxpool2d_nchw_f32_w8a32(y1, n, cv1_c_out, h, w, pool_k, 1, pad, y2, h, w);
    maxpool2d_nchw_f32_w8a32(y2, n, cv1_c_out, h, w, pool_k, 1, pad, y3, h, w);
    yolo_timing_end();

    yolo_timing_begin("concat");
    concat4_nchw_f32_w8a32(x1, cv1_c_out, y1, cv1_c_out, y2, cv1_c_out, y3, cv1_c_out,
                     n, h, w, cat);
    yolo_timing_end();
    yolo_timing_begin("cv2");
    if (cv2_is_int8 && cv2_w) {
        conv2d_nchw_f32_w8_w8a32(cat, n, 4 * cv1_c_out, h, w,
                           (const int8_t*)cv2_w, cv2_scale, cv2_c_out, 1, 1,
                           cv2_bias, 1, 1, 0, 0, 1,
                           y, h, w);
    } else if (cv2_w) {
        conv2d_nchw_f32_w8a32(cat, n, 4 * cv1_c_out, h, w,
                        (const float*)cv2_w, cv2_c_out, 1, 1,
                        cv2_bias, 1, 1, 0, 0, 1,
                        y, h, w);
    }
    silu_nchw_f32_w8a32(y, n, cv2_c_out, h, w, y);
    yolo_timing_end();

    feature_pool_free(cat);
    feature_pool_free(y3);
    feature_pool_free(y2);
    feature_pool_free(y1);
    feature_pool_free(x1);
}
