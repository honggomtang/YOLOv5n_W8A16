#include "c3_w8a16.h"
#include "../operations/conv2d_w8a16.h"
#include "../operations/silu_w8a16.h"
#include "../operations/bottleneck_w8a16.h"
#include "../operations/concat_w8a16.h"
#include "../utils/feature_pool.h"
#include "../utils/timing.h"
#include "../utils/weights_loader.h"
#if defined(USE_CONV_ACC) && defined(BARE_METAL)
#include "../drivers/conv_acc_driver.h"
#endif
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#ifdef BARE_METAL
#include "xil_printf.h"
#endif

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

static int conv1x1_int16_w8a16(
    const int16_t* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const int8_t* w_ptr, int32_t c_out, const int32_t* bias, uint32_t multiplier,
    int16_t* y)
{
#if defined(USE_CONV_ACC) && defined(BARE_METAL)
    int32_t padded_h = h;
    int32_t padded_w = w;
    uint32_t need = conv_acc_scratch_size(c_in, 1, 1, padded_h, padded_w, h, w);
    void* scratch = feature_pool_scratch_alloc((size_t)need);
    if (scratch && need > 0) {
        int acc_used = conv_layer_run(x, n, c_in, h, w, w_ptr, c_out, 1, 1,
            bias, multiplier, 1, 1, 0, 0,
            y, h, w, scratch, need);
        silu_nchw_w8a16(y, n, c_out, h, w, y);
        return acc_used;
    }
#endif
    conv2d_nchw_w8a16(x, n, c_in, h, w, w_ptr, c_out, 1, 1,
                      bias, multiplier, 1, 1, 0, 0, 1,
                      y, h, w);
    silu_nchw_w8a16(y, n, c_out, h, w, y);
    return 0;
}

void c3_nchw_w8a16(
    weights_loader_t* loader,
    const int16_t* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const char* cv1_weight_name, const char* cv2_weight_name, const char* cv3_weight_name,
    int32_t n_bottleneck,
    const char** bn_cv1_weight_names, const char** bn_cv2_weight_names,
    int32_t shortcut,
    int16_t* y)
{
    float s1, s2, s3;
    int i1, i2, i3;
    void* w1 = weights_get_tensor_for_conv(loader, cv1_weight_name, &s1, &i1);
    void* w2 = weights_get_tensor_for_conv(loader, cv2_weight_name, &s2, &i2);
    void* w3 = weights_get_tensor_for_conv(loader, cv3_weight_name, &s3, &i3);
    if (!w1 || !w2 || !w3) return;

    const tensor_info_t* t1 = weights_find_tensor(loader, cv1_weight_name);
    const tensor_info_t* t2 = weights_find_tensor(loader, cv2_weight_name);
    const tensor_info_t* t3 = weights_find_tensor(loader, cv3_weight_name);
    int32_t cv1_c_out = t1 && t1->ndim >= 1 ? t1->shape[0] : 16;
    int32_t cv2_c_out = t2 && t2->ndim >= 1 ? t2->shape[0] : 16;
    int32_t cv3_c_out = t3 && t3->ndim >= 1 ? t3->shape[0] : 32;

    char bias_name[512];
    static int32_t cv1_bias_buf[256], cv2_bias_buf[256], cv3_bias_buf[256];
    weight_name_to_bias_name(cv1_weight_name, bias_name, sizeof(bias_name));
    const float* b1 = weights_get_tensor_data(loader, bias_name);
    bias_convert(b1, s1, cv1_c_out, cv1_bias_buf);
    weight_name_to_bias_name(cv2_weight_name, bias_name, sizeof(bias_name));
    const float* b2 = weights_get_tensor_data(loader, bias_name);
    bias_convert(b2, s2, cv2_c_out, cv2_bias_buf);
    weight_name_to_bias_name(cv3_weight_name, bias_name, sizeof(bias_name));
    const float* b3 = weights_get_tensor_data(loader, bias_name);
    bias_convert(b3, s3, cv3_c_out, cv3_bias_buf);

    uint32_t cv1_mult = scale_to_mult(s1);
    uint32_t cv2_mult = scale_to_mult(s2);
    uint32_t cv3_mult = scale_to_mult(s3);

    const size_t cv1_bytes = (size_t)n * (size_t)cv1_c_out * (size_t)h * (size_t)w * sizeof(int16_t);
    const size_t cv2_bytes = (size_t)n * (size_t)cv2_c_out * (size_t)h * (size_t)w * sizeof(int16_t);
    const size_t cat_bytes = (size_t)n * (size_t)(cv1_c_out + cv2_c_out) * (size_t)h * (size_t)w * sizeof(int16_t);

    int16_t* concat_out = (int16_t*)feature_pool_scratch_alloc(cat_bytes);
    int16_t* cv1_out = (int16_t*)feature_pool_scratch_alloc(cv1_bytes);
    int16_t* cv2_out = (int16_t*)feature_pool_scratch_alloc(cv2_bytes);
    int16_t* bn_a = (int16_t*)feature_pool_scratch_alloc(cv1_bytes);
    int16_t* bn_b = (int16_t*)feature_pool_scratch_alloc(cv1_bytes);

    if (!concat_out || !cv1_out || !cv2_out || !bn_a || !bn_b) {
#ifdef BARE_METAL
        xil_printf("C3 W8A16 scratch alloc failed\n");
#endif
        return;
    }

    yolo_timing_begin("cv1");
    int acc1 = conv1x1_int16_w8a16(x, n, c_in, h, w, (const int8_t*)w1, cv1_c_out, cv1_bias_buf, cv1_mult, cv1_out);
    yolo_timing_end_with_op(acc1 ? "cv1_acc" : "cv1");
    yolo_timing_begin("cv2");
    int acc2 = conv1x1_int16_w8a16(x, n, c_in, h, w, (const int8_t*)w2, cv2_c_out, cv2_bias_buf, cv2_mult, cv2_out);
    yolo_timing_end_with_op(acc2 ? "cv2_acc" : "cv2");
    yolo_timing_begin("bottleneck");
    static int32_t bn_cv1_buf[3][128], bn_cv2_buf[3][128];
    const int16_t* bn_in = cv1_out;
    int16_t* bn_out = bn_a;
    for (int32_t i = 0; i < n_bottleneck; i++) {
        bn_out = (i % 2 == 0) ? bn_a : bn_b;
        float bs1, bs2;
        int ib1, ib2;
        void* bw1 = weights_get_tensor_for_conv(loader, bn_cv1_weight_names[i], &bs1, &ib1);
        void* bw2 = weights_get_tensor_for_conv(loader, bn_cv2_weight_names[i], &bs2, &ib2);
        if (!bw1 || !bw2) break;
        weight_name_to_bias_name(bn_cv1_weight_names[i], bias_name, sizeof(bias_name));
        const float* bb1 = weights_get_tensor_data(loader, bias_name);
        bias_convert(bb1, bs1, cv1_c_out, bn_cv1_buf[i]);
        weight_name_to_bias_name(bn_cv2_weight_names[i], bias_name, sizeof(bias_name));
        const float* bb2 = weights_get_tensor_data(loader, bias_name);
        bias_convert(bb2, bs2, cv1_c_out, bn_cv2_buf[i]);
        uint32_t bn_m1 = scale_to_mult(bs1);
        uint32_t bn_m2 = scale_to_mult(bs2);
        bottleneck_nchw_w8a16(
            bn_in, n, cv1_c_out, h, w,
            (const int8_t*)bw1, cv1_c_out, bn_cv1_buf[i], bn_m1,
            (const int8_t*)bw2, cv1_c_out, bn_cv2_buf[i], bn_m2,
            shortcut,
            bn_out);
        bn_in = bn_out;
    }
    yolo_timing_end();
    yolo_timing_begin("concat");
    concat_nchw_w8a16(bn_out, cv1_c_out, cv2_out, cv2_c_out, n, h, w, concat_out);
    yolo_timing_end();
    yolo_timing_begin("cv3");
    int acc3 = conv1x1_int16_w8a16(concat_out, n, cv1_c_out + cv2_c_out, h, w, (const int8_t*)w3, cv3_c_out, cv3_bias_buf, cv3_mult, y);
    yolo_timing_end_with_op(acc3 ? "cv3_acc" : "cv3");
}

static void conv1x1_w8a16(
    const float* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const void* w_ptr, float w_scale, int w_is_int8, int32_t c_out, const float* bias,
    float* y)
{
    if (w_is_int8) {
        conv2d_nchw_f32_w8_w8a16(x, n, c_in, h, w,
                           (const int8_t*)w_ptr, w_scale, c_out, 1, 1,
                           bias, 1, 1, 0, 0, 1,
                           y, h, w);
    } else {
        conv2d_nchw_f32_w8a16(x, n, c_in, h, w,
                        (const float*)w_ptr, c_out, 1, 1,
                        bias, 1, 1, 0, 0, 1,
                        y, h, w);
    }
    silu_nchw_f32_w8a16(y, n, c_out, h, w, y);
}

void c3_nchw_f32_w8a16(
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
    conv1x1_w8a16(x, n, c_in, h, w, cv1_w, cv1_scale, cv1_is_int8, cv1_c_out, cv1_bias, cv1_out);
    yolo_timing_end();
    yolo_timing_begin("cv2");
    conv1x1_w8a16(x, n, c_in, h, w, cv2_w, cv2_scale, cv2_is_int8, cv2_c_out, cv2_bias, cv2_out);
    yolo_timing_end();
    yolo_timing_begin("bottleneck");
    const float* bn_in = cv1_out;
    float* bn_out = bn_a;
    for (int32_t i = 0; i < n_bottleneck; i++) {
        bn_out = (i % 2 == 0) ? bn_a : bn_b;
        bottleneck_nchw_f32_w8a16(
            bn_in, n, cv1_c_out, h, w,
            bn_cv1_w[i], bn_cv1_scale[i], bn_cv1_is_int8[i], cv1_c_out, bn_cv1_bias[i],
            bn_cv2_w[i], bn_cv2_scale[i], bn_cv2_is_int8[i], cv1_c_out, bn_cv2_bias[i],
            shortcut,
            bn_out);
        bn_in = bn_out;
    }
    yolo_timing_end();
    yolo_timing_begin("concat");
    concat_nchw_f32_w8a16(bn_out, cv1_c_out, cv2_out, cv2_c_out, n, h, w, concat_out);
    yolo_timing_end();
    yolo_timing_begin("cv3");
    conv1x1_w8a16(concat_out, n, cv1_c_out + cv2_c_out, h, w, cv3_w, cv3_scale, cv3_is_int8, cv3_c_out, cv3_bias, y);
    yolo_timing_end();

    feature_pool_free(concat_out);
    feature_pool_free(bn_b);
    feature_pool_free(bn_a);
    feature_pool_free(cv2_out);
    feature_pool_free(cv1_out);
}
