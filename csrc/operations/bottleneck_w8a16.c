#include "bottleneck_w8a16.h"
#include "conv2d_w8a16.h"
#include "silu_w8a16.h"
#include "../utils/feature_pool.h"
#include <stddef.h>
#include <stdint.h>

static inline int16_t clamp_s16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

/* Output = SiLU(Conv3x3(SiLU(Conv1x1(X)))) + X (shortcut 시). 잔차는 int32_t 덧셈 후 clamp_s16. */
void bottleneck_nchw_w8a16(
    const int16_t* x, int32_t n, int32_t c, int32_t h, int32_t w,
    const int8_t* cv1_w, int32_t cv1_c_out, const int32_t* cv1_bias, uint32_t cv1_mult,
    const int8_t* cv2_w, int32_t cv2_c_out, const int32_t* cv2_bias, uint32_t cv2_mult,
    int32_t shortcut,
    int16_t* y)
{
    const size_t cv1_bytes = (size_t)n * (size_t)cv1_c_out * (size_t)h * (size_t)w * sizeof(int16_t);
    const size_t cv2_bytes = (size_t)n * (size_t)cv2_c_out * (size_t)h * (size_t)w * sizeof(int16_t);
    int16_t* cv1_out = (int16_t*)feature_pool_scratch_alloc(cv1_bytes);
    int16_t* cv2_out = (int16_t*)feature_pool_scratch_alloc(cv2_bytes);
    if (!cv1_out || !cv2_out)
        return;

    /* Conv1x1 -> SiLU */
    conv2d_nchw_w8a16(x, n, c, h, w, cv1_w, cv1_c_out, 1, 1,
                      cv1_bias, cv1_mult, 1, 1, 0, 0, 1,
                      cv1_out, h, w);
    silu_nchw_w8a16(cv1_out, n, cv1_c_out, h, w, cv1_out);

    /* Conv3x3 -> SiLU */
    conv2d_nchw_w8a16(cv1_out, n, cv1_c_out, h, w, cv2_w, cv2_c_out, 3, 3,
                      cv2_bias, cv2_mult, 1, 1, 1, 1, 1,
                      cv2_out, h, w);
    silu_nchw_w8a16(cv2_out, n, cv2_c_out, h, w, cv2_out);

    if (shortcut && c == cv2_c_out) {
        const int32_t size = n * c * h * w;
        for (int32_t i = 0; i < size; i++) {
            int32_t sum = (int32_t)x[i] + (int32_t)cv2_out[i];
            y[i] = clamp_s16(sum);
        }
    } else {
        const int32_t size = n * cv2_c_out * h * w;
        for (int32_t i = 0; i < size; i++)
            y[i] = cv2_out[i];
    }

    /* scratch 사용 시 free 없음; 추론 시작 시 feature_pool_scratch_reset() 호출 전제 */
}

void bottleneck_nchw_f32_w8a16(
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
        conv2d_nchw_f32_w8_w8a16(x, n, c, h, w,
                           (const int8_t*)cv1_w, cv1_scale, cv1_c_out, 1, 1,
                           cv1_bias, 1, 1, 0, 0, 1,
                           cv1_out, h, w);
    } else {
        conv2d_nchw_f32_w8a16(x, n, c, h, w,
                        (const float*)cv1_w, cv1_c_out, 1, 1,
                        cv1_bias, 1, 1, 0, 0, 1,
                        cv1_out, h, w);
    }
    silu_nchw_f32_w8a16(cv1_out, n, cv1_c_out, h, w, cv1_out);
    if (cv2_is_int8) {
        conv2d_nchw_f32_w8_w8a16(cv1_out, n, cv1_c_out, h, w,
                           (const int8_t*)cv2_w, cv2_scale, cv2_c_out, 3, 3,
                           cv2_bias, 1, 1, 1, 1, 1,
                           cv2_out, h, w);
    } else {
        conv2d_nchw_f32_w8a16(cv1_out, n, cv1_c_out, h, w,
                        (const float*)cv2_w, cv2_c_out, 3, 3,
                        cv2_bias, 1, 1, 1, 1, 1,
                        cv2_out, h, w);
    }
    silu_nchw_f32_w8a16(cv2_out, n, cv2_c_out, h, w, cv2_out);
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
