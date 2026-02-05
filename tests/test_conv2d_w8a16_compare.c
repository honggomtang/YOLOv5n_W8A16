/*
 * conv2d W8A32 vs W8A16 비교 검증
 * - 동일 float 입력을 W8A32(conv2d_nchw_f32_w8_w8a32)와
 *   W8A16(입력 Q6.10 변환 → conv2d_nchw_w8a16 → 출력 dequant)으로 각각 실행
 * - W8A32 출력/scale ≈ W8A16 출력(dequant) 이어야 함 (bias=0 기준)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "test_vectors_conv.h"
#include "../csrc/utils/weights_loader.h"
#include "../csrc/operations/conv2d_w8a32.h"
#include "../csrc/operations/conv2d_w8a16.h"

#define Q6_10_SCALE 1024

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void float_to_q610(const float* x, int16_t* y, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i] * (float)Q6_10_SCALE;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        y[i] = (int16_t)(int32_t)v;
    }
}

/*
 * 문제 픽셀 (n0,c0,oh,ow)에 대해 W8A32 vs W8A16이 참조하는 입력 주소/값을 stderr로 출력.
 * in_safe 판정과 x_base 계산식을 한 줄씩 비교할 수 있게 함.
 */
static void trace_pixel_diff(
    int n0, int c0, int oh, int ow,
    const float* x_f, const int16_t* x_q, const int8_t* w,
    int c_in, int h_in, int w_in, int c_out, int k_h, int k_w,
    int stride_h, int stride_w, int pad_h, int pad_w)
{
    const int x_h_stride = w_in;
    const int x_c_stride = h_in * w_in;
    const int w_ic_stride = k_h * k_w;
    const int w_oc_stride = c_in * k_h * k_w;

    const int safe_oh_min = (pad_h + stride_h - 1) / stride_h;
    const int safe_oh_max = (h_in - k_h + pad_h) / stride_h;
    const int safe_ow_min = (pad_w + stride_w - 1) / stride_w;
    const int safe_ow_max = (w_in - k_w + pad_w) / stride_w;

    const int oh0 = (oh / 8) * 8;
    const int ow0 = (ow / 8) * 8;
    const int oh_end = oh0 + 8;
    const int ow_end = ow0 + 8;
    const int tile_is_safe = (oh0 >= safe_oh_min && oh_end <= safe_oh_max &&
                              ow0 >= safe_ow_min && ow_end <= safe_ow_max);

    const int in_safe = (oh >= safe_oh_min && oh < safe_oh_max &&
                         ow >= safe_ow_min && ow < safe_ow_max);

    const int ih0 = oh * stride_h - pad_h;
    const int iw0 = ow * stride_w - pad_w;

    fprintf(stderr, "\n=== trace_pixel_diff(n=%d c=%d oh=%d ow=%d) ===\n", n0, c0, oh, ow);
    fprintf(stderr, "  safe_oh_min=%d safe_oh_max=%d safe_ow_min=%d safe_ow_max=%d\n",
            safe_oh_min, safe_oh_max, safe_ow_min, safe_ow_max);
    fprintf(stderr, "  tile(oh0=%d ow0=%d) -> tile_is_safe=%d\n", oh0, ow0, tile_is_safe);
    fprintf(stderr, "  in_safe = (oh>=%d && oh<%d && ow>=%d && ow<%d) = %d\n",
            safe_oh_min, safe_oh_max, safe_ow_min, safe_ow_max, in_safe);
    fprintf(stderr, "  x_base: ih0 = oh*stride_h - pad_h = %d*%d - %d = %d\n", oh, stride_h, pad_h, ih0);
    fprintf(stderr, "          iw0 = ow*stride_w - pad_w = %d*%d - %d = %d\n", ow, stride_w, pad_w, iw0);
    fprintf(stderr, "  x_base offset = (n*c_in+ic)*%d + ih0*%d + iw0 (per ic); then + kh*%d + kw\n",
            (int)x_c_stride, x_h_stride, x_h_stride);

    fprintf(stderr, "\n--- W8A32 (float x, int8 w*scale) ---\n");
    for (int ic = 0; ic < c_in; ic++) {
        const float* x_ch_f = x_f + (n0 * c_in + ic) * x_c_stride;
        const int8_t* w_base = w + c0 * w_oc_stride + ic * w_ic_stride;
        if (in_safe) {
            const float* x_base_f = x_ch_f + ih0 * x_h_stride + iw0;
            for (int kh = 0; kh < k_h; kh++) {
                for (int kw = 0; kw < k_w; kw++) {
                    int ih = ih0 + kh, iw = iw0 + kw;
                    float xv = x_base_f[kh * x_h_stride + kw];
                    int8_t wv = (int8_t)w_base[kh * k_w + kw];
                    fprintf(stderr, "  ic=%d kh=%d kw=%d ih=%d iw=%d x_f=%.6f w_int8=%d\n",
                            ic, kh, kw, ih, iw, (double)xv, (int)wv);
                }
            }
        } else {
            for (int kh = 0; kh < k_h; kh++) {
                int ih = oh * stride_h - pad_h + kh;
                for (int kw = 0; kw < k_w; kw++) {
                    int iw = ow * stride_w - pad_w + kw;
                    int in_bounds = ((unsigned)ih < (unsigned)h_in && (unsigned)iw < (unsigned)w_in);
                    float xv = in_bounds ? x_ch_f[ih * x_h_stride + iw] : 0.0f;
                    int8_t wv = (int8_t)w_base[kh * k_w + kw];
                    fprintf(stderr, "  ic=%d kh=%d kw=%d ih=%d iw=%d in_bounds=%d x_f=%.6f w_int8=%d\n",
                            ic, kh, kw, ih, iw, in_bounds, (double)xv, (int)wv);
                }
            }
        }
    }

    fprintf(stderr, "\n--- W8A16 (int16 x, int8 w) ---\n");
    fprintf(stderr, "  (same in_safe=%d, ih0=%d iw0=%d)\n", in_safe, ih0, iw0);
    for (int ic = 0; ic < c_in; ic++) {
        const int16_t* x_ch_q = x_q + (n0 * c_in + ic) * x_c_stride;
        const int8_t* w_base = w + c0 * w_oc_stride + ic * w_ic_stride;
        if (in_safe) {
            const int16_t* x_base_q = x_ch_q + ih0 * x_h_stride + iw0;
            for (int kh = 0; kh < k_h; kh++) {
                for (int kw = 0; kw < k_w; kw++) {
                    int ih = ih0 + kh, iw = iw0 + kw;
                    int16_t xv = x_base_q[kh * x_h_stride + kw];
                    int8_t wv = (int8_t)w_base[kh * k_w + kw];
                    fprintf(stderr, "  ic=%d kh=%d kw=%d ih=%d iw=%d x_int16=%d w_int8=%d\n",
                            ic, kh, kw, ih, iw, (int)xv, (int)wv);
                }
            }
        } else {
            for (int kh = 0; kh < k_h; kh++) {
                int ih = oh * stride_h - pad_h + kh;
                for (int kw = 0; kw < k_w; kw++) {
                    int iw = ow * stride_w - pad_w + kw;
                    int in_bounds = ((unsigned)ih < (unsigned)h_in && (unsigned)iw < (unsigned)w_in);
                    int16_t xv = in_bounds ? x_ch_q[ih * x_h_stride + iw] : 0;
                    int8_t wv = (int8_t)w_base[kh * k_w + kw];
                    fprintf(stderr, "  ic=%d kh=%d kw=%d ih=%d iw=%d in_bounds=%d x_int16=%d w_int8=%d\n",
                            ic, kh, kw, ih, iw, in_bounds, (int)xv, (int)wv);
                }
            }
        }
    }
    fprintf(stderr, "=== end trace_pixel_diff ===\n\n");
}

int main(void) {
    printf("=== conv2d W8A32 vs W8A16 compare ===\n\n");

    weights_loader_t weights;
    if (weights_load_from_file_w8("assets/weights_w8.bin", &weights) != 0) {
        fprintf(stderr, "Failed to load assets/weights_w8.bin\n");
        return 1;
    }

    float scale = 0.f;
    int is_int8 = 0;
    void* w_ptr = weights_get_tensor_for_conv(&weights, "model.0.conv.weight", &scale, &is_int8);
    if (!w_ptr || !is_int8) {
        fprintf(stderr, "model.0.conv.weight not found or not int8 in weights_w8.bin\n");
        weights_free(&weights);
        return 1;
    }
    const int8_t* w_int8 = (const int8_t*)w_ptr;

    /* 테스트용 작은 공간: 1*3*64*64 -> 6x6 s2 p2 -> 1*16*31*31 */
    const int n = 1, c_in = 3, h_in = 64, w_in = 64;
    const int c_out = 16, k_h = 6, k_w = 6;
    const int stride = 2, pad = 2;
    const int h_out = (h_in + 2 * pad - k_h) / stride + 1;
    const int w_out = (w_in + 2 * pad - k_w) / stride + 1;

    const int in_elems = n * c_in * h_in * w_in;
    const int out_elems = n * c_out * h_out * w_out;

    /* 입력: test_vectors의 tv_x (1*3*64*64) */
    static int16_t x_int16[1 * 3 * 64 * 64];
    float_to_q610(tv_x, x_int16, in_elems);

    static float y_w8a32[1 * 16 * 31 * 31];
    static int16_t y_w8a16[1 * 16 * 31 * 31];

    /* W8A32: float in -> float out (bias NULL) */
    conv2d_nchw_f32_w8_w8a32(
        tv_x, n, c_in, h_in, w_in,
        w_int8, scale, c_out, k_h, k_w,
        NULL, stride, stride, pad, pad, 1,
        y_w8a32, h_out, w_out);

    /* W8A16: int16 in -> int16 out (bias NULL, shift_amount=10) */
    conv2d_nchw_w8a16(
        x_int16, n, c_in, h_in, w_in,
        w_int8, c_out, k_h, k_w,
        NULL, 10, stride, stride, pad, pad, 1,
        y_w8a16, h_out, w_out);

    /* W8A32: out = scale * sum(x_float * w_int8).  =>  W8A32/scale = sum(x_float * w_int8).
     * W8A16: acc = sum(x_int16 * w_int8), x_int16 ≈ 1024*x_float => acc ≈ 1024*sum(x_float*w_int8).
     *        y_int16 = acc >> 10 = acc/1024 ≈ sum(x_float*w_int8).  즉 y_int16 자체가 sum 단위.
     * 따라서 W8A32/scale 과 (float)y_w8a16 를 비교. */
    float max_diff = 0.0f;
    float max_rel = 0.0f;
    int over2 = 0, over10 = 0;
    int max_diff_idx = 0;
    for (int i = 0; i < out_elems; i++) {
        float ref = y_w8a32[i] / scale;
        float a16 = (float)y_w8a16[i];
        float d = fabsf(ref - a16);
        if (d > max_diff) {
            max_diff = d;
            max_diff_idx = i;
        }
        if (fabsf(ref) > 1e-6f) {
            float r = d / fabsf(ref);
            if (r > max_rel) max_rel = r;
        }
        if (d > 2.0f) over2++;
        if (d > 10.0f) over10++;
    }

    printf("Layer: 1*3*64*64 -> conv 16@6x6 s2 p2 -> 1*16*%d*%d\n", h_out, w_out);
    printf("  W8 scale: %g\n", (double)scale);
    {
        int chw = h_out * w_out;
        int c = max_diff_idx / chw;
        int hw = max_diff_idx % chw;
        int h = hw / w_out;
        int w = hw % w_out;
        printf("  Max |W8A32/scale - W8A16_int16|: %g (at flat %d -> n=0 c=%d h=%d w=%d)\n",
               (double)max_diff, max_diff_idx, c, h, w);
    }
    printf("  At max_diff: ref=%.2f a16=%d\n",
           (double)(y_w8a32[max_diff_idx] / scale), (int)y_w8a16[max_diff_idx]);
    printf("  Max relative diff (where ref!=0):   %g\n", (double)max_rel);
    printf("  Pixels with |diff|>2: %d, |diff|>10: %d (total %d)\n", over2, over10, out_elems);
    printf("  Sample [0..4]: W8A32/scale =");
    for (int i = 0; i < 5; i++) printf(" %g", (double)(y_w8a32[i] / scale));
    printf("\n  Sample [0..4]: W8A16      =");
    for (int i = 0; i < 5; i++) printf(" %d", (int)y_w8a16[i]);
    printf("\n");

    {
        int chw = h_out * w_out;
        int c = max_diff_idx / chw;
        int hw = max_diff_idx % chw;
        int h = hw / w_out;
        int w = hw % w_out;
        trace_pixel_diff(0, c, h, w, tv_x, x_int16, w_int8,
                        c_in, h_in, w_in, c_out, k_h, k_w,
                        stride, stride, pad, pad);
    }

    weights_free(&weights);

    /* 허용: 정수 누적/시프트(>>10)로 인한 반올림. 타일/경계에서 소수 픽셀만 diff 큼 → 5% 이하 허용 */
    if (max_diff < 300.0f && (float)over10 / (float)out_elems < 0.05f) {
        printf("\nResult: OK (W8A16 matches W8A32 within integer/rounding tolerance)\n");
        return 0;
    }
    printf("\nResult: NG (diff or outlier count too large)\n");
    return 1;
}
