#include "bottleneck_w8a32.h"
#include "conv2d_w8a32.h"
#include "silu_w8a32.h"
#include "../utils/feature_pool.h"

void bottleneck_nchw_f32_w8a32(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    const void* cv1_w, float cv1_scale, int cv1_is_int8, int32_t cv1_c_out, const float* cv1_bias,
    const void* cv2_w, float cv2_scale, int cv2_is_int8, int32_t cv2_c_out, const float* cv2_bias,
    int32_t shortcut,
    float* y)
{
    size_t cv1_bytes = (size_t)n * (size_t)cv1_c_out * (size_t)h * (size_t)w * sizeof(float);
    size_t cv2_bytes = (size_t)n * (size_t)cv2_c_out * (size_t)h * (size_t)w * sizeof(float);
    float* cv1_out = (float*)feature_pool_alloc(cv1_bytes);
    float* cv2_out = (float*)feature_pool_alloc(cv2_bytes);
    if (!cv1_out || !cv2_out) {
        if (cv2_out) feature_pool_free(cv2_out);
        if (cv1_out) feature_pool_free(cv1_out);
        return;
    }

    if (cv1_is_int8) {
        conv2d_nchw_f32_w8_w8a32(x, n, c, h, w,
                           (const int8_t*)cv1_w, cv1_scale, cv1_c_out, 1, 1,
                           cv1_bias, 1, 1, 0, 0, 1,
                           cv1_out, h, w);
    } else {
        conv2d_nchw_f32_w8a32(x, n, c, h, w,
                        (const float*)cv1_w, cv1_c_out, 1, 1,
                        cv1_bias, 1, 1, 0, 0, 1,
                        cv1_out, h, w);
    }
    silu_nchw_f32_w8a32(cv1_out, n, cv1_c_out, h, w, cv1_out);
    if (cv2_is_int8) {
        conv2d_nchw_f32_w8_w8a32(cv1_out, n, cv1_c_out, h, w,
                           (const int8_t*)cv2_w, cv2_scale, cv2_c_out, 3, 3,
                           cv2_bias, 1, 1, 1, 1, 1,
                           cv2_out, h, w);
    } else {
        conv2d_nchw_f32_w8a32(cv1_out, n, cv1_c_out, h, w,
                        (const float*)cv2_w, cv2_c_out, 3, 3,
                        cv2_bias, 1, 1, 1, 1, 1,
                        cv2_out, h, w);
    }
    silu_nchw_f32_w8a32(cv2_out, n, cv2_c_out, h, w, cv2_out);
    if (shortcut && c == cv2_c_out) {
        int32_t size = n * c * h * w;
        for (int32_t i = 0; i < size; i++) {
            y[i] = x[i] + cv2_out[i];
        }
    } else {
        int32_t size = n * cv2_c_out * h * w;
        for (int32_t i = 0; i < size; i++) {
            y[i] = cv2_out[i];
        }
    }

    feature_pool_free(cv2_out);
    feature_pool_free(cv1_out);
}
