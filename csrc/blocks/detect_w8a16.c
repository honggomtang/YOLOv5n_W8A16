#include "detect_w8a16.h"
#include "../operations/conv2d_w8a16.h"
#include "../utils/timing.h"
#include "../utils/weights_loader.h"
#include <string.h>
#include <math.h>

void detect_nchw_f32_w8a16(
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
        conv2d_nchw_f32_w8_w8a16(p3, 1, p3_c, p3_h, p3_w,
            (const int8_t*)m0_w, m0_scale, 255, 1, 1, m0_b, 1, 1, 0, 0, 1,
            p3_out, p3_h, p3_w);
    } else {
        conv2d_nchw_f32_w8a16(p3, 1, p3_c, p3_h, p3_w,
            (const float*)m0_w, 255, 1, 1, m0_b, 1, 1, 0, 0, 1,
            p3_out, p3_h, p3_w);
    }
    if (m1_is_int8) {
        conv2d_nchw_f32_w8_w8a16(p4, 1, p4_c, p4_h, p4_w,
            (const int8_t*)m1_w, m1_scale, 255, 1, 1, m1_b, 1, 1, 0, 0, 1,
            p4_out, p4_h, p4_w);
    } else {
        conv2d_nchw_f32_w8a16(p4, 1, p4_c, p4_h, p4_w,
            (const float*)m1_w, 255, 1, 1, m1_b, 1, 1, 0, 0, 1,
            p4_out, p4_h, p4_w);
    }
    if (m2_is_int8) {
        conv2d_nchw_f32_w8_w8a16(p5, 1, p5_c, p5_h, p5_w,
            (const int8_t*)m2_w, m2_scale, 255, 1, 1, m2_b, 1, 1, 0, 0, 1,
            p5_out, p5_h, p5_w);
    } else {
        conv2d_nchw_f32_w8a16(p5, 1, p5_c, p5_h, p5_w,
            (const float*)m2_w, 255, 1, 1, m2_b, 1, 1, 0, 0, 1,
            p5_out, p5_h, p5_w);
    }
    yolo_timing_end();
}

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

void detect_nchw_w8a16(
    weights_loader_t* loader,
    const int16_t* p3, int32_t p3_c, int32_t p3_h, int32_t p3_w,
    const int16_t* p4, int32_t p4_c, int32_t p4_h, int32_t p4_w,
    const int16_t* p5, int32_t p5_c, int32_t p5_h, int32_t p5_w,
    const char* m0_weight_name, const char* m1_weight_name, const char* m2_weight_name,
    int32_t c_detect,
    int16_t* p3_out, int16_t* p4_out, int16_t* p5_out)
{
    float s0, s1, s2;
    int i0, i1, i2;
    void* w0 = weights_get_tensor_for_conv(loader, m0_weight_name, &s0, &i0);
    void* w1 = weights_get_tensor_for_conv(loader, m1_weight_name, &s1, &i1);
    void* w2 = weights_get_tensor_for_conv(loader, m2_weight_name, &s2, &i2);
    if (!w0 || !w1 || !w2) return;

    char bias_name[256];
    static int32_t m0_bias_buf[256], m1_bias_buf[256], m2_bias_buf[256];
    weight_name_to_bias_name(m0_weight_name, bias_name, sizeof(bias_name));
    bias_convert(weights_get_tensor_data(loader, bias_name), s0, c_detect, m0_bias_buf);
    weight_name_to_bias_name(m1_weight_name, bias_name, sizeof(bias_name));
    bias_convert(weights_get_tensor_data(loader, bias_name), s1, c_detect, m1_bias_buf);
    weight_name_to_bias_name(m2_weight_name, bias_name, sizeof(bias_name));
    bias_convert(weights_get_tensor_data(loader, bias_name), s2, c_detect, m2_bias_buf);

    uint32_t m0_mult = scale_to_mult(s0);
    uint32_t m1_mult = scale_to_mult(s1);
    uint32_t m2_mult = scale_to_mult(s2);

    yolo_timing_begin("detect");
    conv2d_nchw_w8a16(p3, 1, p3_c, p3_h, p3_w, (const int8_t*)w0, c_detect, 1, 1,
                     m0_bias_buf, m0_mult, 1, 1, 0, 0, 1,
                     p3_out, p3_h, p3_w);
    conv2d_nchw_w8a16(p4, 1, p4_c, p4_h, p4_w, (const int8_t*)w1, c_detect, 1, 1,
                     m1_bias_buf, m1_mult, 1, 1, 0, 0, 1,
                     p4_out, p4_h, p4_w);
    conv2d_nchw_w8a16(p5, 1, p5_c, p5_h, p5_w, (const int8_t*)w2, c_detect, 1, 1,
                     m2_bias_buf, m2_mult, 1, 1, 0, 0, 1,
                     p5_out, p5_h, p5_w);
    yolo_timing_end();
}
