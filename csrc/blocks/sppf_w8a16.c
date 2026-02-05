#include "sppf_w8a16.h"
#include "../operations/conv2d_w8a16.h"
#include "../operations/silu_w8a16.h"
#include "../operations/maxpool2d_w8a16.h"
#include "../operations/concat_w8a16.h"
#include "../utils/feature_pool.h"
#include "../utils/timing.h"
#include "../utils/weights_loader.h"
#include <stddef.h>
#include <string.h>
#include <math.h>

static inline uint32_t scale_to_mult(float s) {
    if (s <= 0.f) return 1U;
    uint32_t u = (uint32_t)(s * 65536.0f + 0.5f);
    return (u < 1) ? 1U : u;
}

static void weight_name_to_bias_name(const char* weight_name, char* bias_buf, size_t buf_size) {
    size_t len = strlen(weight_name);
    if (len >= 7 && len + 1 <= buf_size && strcmp(weight_name + len - 7, ".weight") == 0) {
        size_t prefix_len = len - 7;
        memcpy(bias_buf, weight_name, prefix_len);
        memcpy(bias_buf + prefix_len, ".bias", 6);
    } else {
        bias_buf[0] = '\0';
    }
}

static void bias_convert(const float* b, float scale, int c_out, int32_t* out) {
    if (!b || scale <= 0.f) { for (int k = 0; k < c_out; k++) out[k] = 0; return; }
    float factor = 1024.0f / scale;
    for (int k = 0; k < c_out; k++)
        out[k] = (int32_t)roundf(b[k] * factor);
}

static void sppf_nchw_w8a16_core(
    const int16_t* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const int8_t* cv1_w, int32_t cv1_c_out, const int32_t* cv1_bias, uint32_t cv1_mult,
    const int8_t* cv2_w, int32_t cv2_c_out, const int32_t* cv2_bias, uint32_t cv2_mult,
    int32_t pool_k,
    int16_t* y)
{
    const int32_t pad = pool_k / 2;
    const size_t x1_bytes = (size_t)n * (size_t)cv1_c_out * (size_t)h * (size_t)w * sizeof(int16_t);
    const size_t cat_bytes = (size_t)n * (size_t)(4 * cv1_c_out) * (size_t)h * (size_t)w * sizeof(int16_t);

    int16_t* x1 = (int16_t*)feature_pool_scratch_alloc(x1_bytes);
    int16_t* y1 = (int16_t*)feature_pool_scratch_alloc(x1_bytes);
    int16_t* y2 = (int16_t*)feature_pool_scratch_alloc(x1_bytes);
    int16_t* y3 = (int16_t*)feature_pool_scratch_alloc(x1_bytes);
    int16_t* cat = (int16_t*)feature_pool_scratch_alloc(cat_bytes);

    if (!x1 || !y1 || !y2 || !y3 || !cat)
        return;

    yolo_timing_begin("cv1");
    conv2d_nchw_w8a16(x, n, c_in, h, w, cv1_w, cv1_c_out, 1, 1,
                      cv1_bias, cv1_mult, 1, 1, 0, 0, 1,
                      x1, h, w);
    silu_nchw_w8a16(x1, n, cv1_c_out, h, w, x1);
    yolo_timing_end();

    yolo_timing_begin("maxpool");
    maxpool2d_nchw_w8a16(x1, n, cv1_c_out, h, w, pool_k, 1, pad, y1, h, w);
    maxpool2d_nchw_w8a16(y1, n, cv1_c_out, h, w, pool_k, 1, pad, y2, h, w);
    maxpool2d_nchw_w8a16(y2, n, cv1_c_out, h, w, pool_k, 1, pad, y3, h, w);
    yolo_timing_end();

    yolo_timing_begin("concat");
    concat4_nchw_w8a16(x1, cv1_c_out, y1, cv1_c_out, y2, cv1_c_out, y3, cv1_c_out,
                       n, h, w, cat);
    yolo_timing_end();

    yolo_timing_begin("cv2");
    conv2d_nchw_w8a16(cat, n, 4 * cv1_c_out, h, w, cv2_w, cv2_c_out, 1, 1,
                      cv2_bias, cv2_mult, 1, 1, 0, 0, 1,
                      y, h, w);
    silu_nchw_w8a16(y, n, cv2_c_out, h, w, y);
    yolo_timing_end();
}

void sppf_nchw_w8a16(
    weights_loader_t* loader,
    const int16_t* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const char* cv1_weight_name, const char* cv2_weight_name,
    int32_t pool_k,
    int16_t* y)
{
    float s1, s2;
    int i1, i2;
    void* w1 = weights_get_tensor_for_conv(loader, cv1_weight_name, &s1, &i1);
    void* w2 = weights_get_tensor_for_conv(loader, cv2_weight_name, &s2, &i2);
    if (!w1 || !w2) return;

    const tensor_info_t* t1 = weights_find_tensor(loader, cv1_weight_name);
    const tensor_info_t* t2 = weights_find_tensor(loader, cv2_weight_name);
    int32_t cv1_c_out = t1 && t1->ndim >= 1 ? t1->shape[0] : 128;
    int32_t cv2_c_out = t2 && t2->ndim >= 1 ? t2->shape[0] : 256;

    char bias_name[512];
    static int32_t cv1_bias_buf[256], cv2_bias_buf[256];
    weight_name_to_bias_name(cv1_weight_name, bias_name, sizeof(bias_name));
    const float* b1 = weights_get_tensor_data(loader, bias_name);
    bias_convert(b1, s1, cv1_c_out, cv1_bias_buf);
    weight_name_to_bias_name(cv2_weight_name, bias_name, sizeof(bias_name));
    const float* b2 = weights_get_tensor_data(loader, bias_name);
    bias_convert(b2, s2, cv2_c_out, cv2_bias_buf);

    uint32_t cv1_mult = scale_to_mult(s1);
    uint32_t cv2_mult = scale_to_mult(s2);

    sppf_nchw_w8a16_core(x, n, c_in, h, w,
        (const int8_t*)w1, cv1_c_out, cv1_bias_buf, cv1_mult,
        (const int8_t*)w2, cv2_c_out, cv2_bias_buf, cv2_mult,
        pool_k, y);
}

void sppf_nchw_f32_w8a16(
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
        conv2d_nchw_f32_w8_w8a16(x, n, c_in, h, w,
                           (const int8_t*)cv1_w, cv1_scale, cv1_c_out, 1, 1,
                           cv1_bias, 1, 1, 0, 0, 1,
                           x1, h, w);
    } else if (cv1_w) {
        conv2d_nchw_f32_w8a16(x, n, c_in, h, w,
                        (const float*)cv1_w, cv1_c_out, 1, 1,
                        cv1_bias, 1, 1, 0, 0, 1,
                        x1, h, w);
    }
    silu_nchw_f32_w8a16(x1, n, cv1_c_out, h, w, x1);
    yolo_timing_end();

    yolo_timing_begin("maxpool");
    maxpool2d_nchw_f32_w8a16(x1, n, cv1_c_out, h, w, pool_k, 1, pad, y1, h, w);
    maxpool2d_nchw_f32_w8a16(y1, n, cv1_c_out, h, w, pool_k, 1, pad, y2, h, w);
    maxpool2d_nchw_f32_w8a16(y2, n, cv1_c_out, h, w, pool_k, 1, pad, y3, h, w);
    yolo_timing_end();

    yolo_timing_begin("concat");
    concat4_nchw_f32_w8a16(x1, cv1_c_out, y1, cv1_c_out, y2, cv1_c_out, y3, cv1_c_out,
                     n, h, w, cat);
    yolo_timing_end();
    yolo_timing_begin("cv2");
    if (cv2_is_int8 && cv2_w) {
        conv2d_nchw_f32_w8_w8a16(cat, n, 4 * cv1_c_out, h, w,
                           (const int8_t*)cv2_w, cv2_scale, cv2_c_out, 1, 1,
                           cv2_bias, 1, 1, 0, 0, 1,
                           y, h, w);
    } else if (cv2_w) {
        conv2d_nchw_f32_w8a16(cat, n, 4 * cv1_c_out, h, w,
                        (const float*)cv2_w, cv2_c_out, 1, 1,
                        cv2_bias, 1, 1, 0, 0, 1,
                        y, h, w);
    }
    silu_nchw_f32_w8a16(y, n, cv2_c_out, h, w, y);
    yolo_timing_end();

    feature_pool_free(cat);
    feature_pool_free(y3);
    feature_pool_free(y2);
    feature_pool_free(y1);
    feature_pool_free(x1);
}
