/*
 * [Conv2D -> Rounding Shift -> SiLU] W8A32 vs W8A16 비교 (Q6.10 고정소수점 검증)
 * - L1은 sum이 31.9를 넘어 Q6.10 범위를 벗어나므로 shift_amount=12 사용 (46 → ~11.5 수준).
 * - W8A16 conv 출력을 Q6.10으로 해석: ref_float = SiLU(conv_out/1024), ref_q = round(ref_float*1024).
 * - W8A16 SiLU LUT 출력과 ref_q(Q6.10 양자화 참조) 비교.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "test_vectors_conv.h"
#include "../csrc/utils/weights_loader.h"
#include "../csrc/operations/conv2d_w8a16.h"
#include "../csrc/operations/silu_w8a16.h"

#define Q6_10_SCALE 1024
/* L1: sum이 46 등으로 Q6.10(±31.9)을 넘으므로 shift_amount=12로 결과를 INT16 Q6.10 범위 안으로 */
#define L1_SHIFT_AMOUNT 12

static void float_to_q610(const float* x, int16_t* y, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i] * (float)Q6_10_SCALE;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        y[i] = (int16_t)(int32_t)v;
    }
}

int main(void) {
    printf("=== Conv2D -> SiLU: W8A32 vs W8A16 compare ===\n\n");

    weights_loader_t weights;
    if (weights_load_from_file_w8("assets/weights_w8.bin", &weights) != 0) {
        fprintf(stderr, "Failed to load assets/weights_w8.bin\n");
        return 1;
    }

    float scale = 0.f;
    int is_int8 = 0;
    void* w_ptr = weights_get_tensor_for_conv(&weights, "model.0.conv.weight", &scale, &is_int8);
    if (!w_ptr || !is_int8) {
        fprintf(stderr, "model.0.conv.weight not found or not int8\n");
        weights_free(&weights);
        return 1;
    }
    const int8_t* w_int8 = (const int8_t*)w_ptr;

    const int n = 1, c_in = 3, h_in = 64, w_in = 64;
    const int c_out = 16, k_h = 6, k_w = 6;
    const int stride = 2, pad = 2;
    const int h_out = (h_in + 2 * pad - k_h) / stride + 1;
    const int w_out = (w_in + 2 * pad - k_w) / stride + 1;
    const int out_elems = n * c_out * h_out * w_out;
    const int in_elems = n * c_in * h_in * w_in;

    static int16_t x_int16[1 * 3 * 64 * 64];
    float_to_q610(tv_x, x_int16, in_elems);

    static int16_t conv_out_w8a16[1 * 16 * 32 * 32];  /* conv만 (Q6.10) */
    static int16_t y_w8a16[1 * 16 * 32 * 32];         /* conv + SiLU LUT (Q6.10) */

    /* W8A16: conv만 (L1은 shift=12로 Q6.10 범위 안으로) → conv_out_w8a16 (Q6.10) */
    conv2d_nchw_w8a16(
        x_int16, n, c_in, h_in, w_in,
        w_int8, c_out, k_h, k_w,
        NULL, L1_SHIFT_AMOUNT, stride, stride, pad, pad, 1,
        conv_out_w8a16, h_out, w_out);

    /* SiLU LUT: Q6.10 → Q6.10 */
    silu_nchw_w8a16(conv_out_w8a16, n, c_out, h_out, w_out, y_w8a16);

    /* 참조: conv_out을 Q6.10으로 해석 → ref_float = SiLU(conv_out/1024), ref_q = round(ref_float*1024) */
    float max_diff = 0.0f;
    float max_rel = 0.0f;
    int over1 = 0, over_half = 0;
    for (int i = 0; i < out_elems; i++) {
        float x_q = (float)conv_out_w8a16[i] / (float)Q6_10_SCALE;
        float ref_float;
        if (x_q >= -100.0f && x_q <= 100.0f)
            ref_float = (float)(x_q / (1.0 + exp(-x_q)));  /* SiLU(x) */
        else
            ref_float = (x_q > 0.0f) ? x_q : 0.0f;
        int32_t ref_q = (int32_t)(ref_float >= 0.0f ? ref_float * Q6_10_SCALE + 0.5f : ref_float * Q6_10_SCALE - 0.5f);
        if (ref_q > 32767) ref_q = 32767;
        if (ref_q < -32768) ref_q = -32768;
        int16_t ref_s16 = (int16_t)ref_q;
        int diff = (int)y_w8a16[i] - (int)ref_s16;
        if (diff < 0) diff = -diff;
        float d = (float)diff;
        if (d > max_diff) max_diff = d;
        if (ref_q != 0) {
            float r = d / (float)(ref_q >= 0 ? ref_q : -ref_q);
            if (r > max_rel) max_rel = r;
        }
        if (diff > 1) over1++;
        if (diff > 512) over_half++;  /* Q6.10에서 0.5 이상 오차 */
    }

    printf("Layer: 1*3*64*64 -> [conv 16@6x6 s2 p2 (shift=%d) + SiLU LUT] -> 1*16*%d*%d\n", L1_SHIFT_AMOUNT, h_out, w_out);
    printf("  Compare: Q6.10 quantized ref (SiLU(conv_out/1024)*1024) vs W8A16 SiLU LUT output\n");
    printf("  Max |ref_q - W8A16|: %.0f (in Q6.10 LSB)\n", (double)max_diff);
    printf("  Max relative diff (ref_q!=0): %g\n", (double)max_rel);
    printf("  Pixels |diff|>1: %d, |diff|>512: %d (total %d)\n", over1, over_half, out_elems);
    printf("  Sample [0..4]: conv_out(Q6.10) =");
    for (int i = 0; i < 5; i++) printf(" %d", (int)conv_out_w8a16[i]);
    printf("\n  Sample [0..4]: W8A16 SiLU   =");
    for (int i = 0; i < 5; i++) printf(" %d", (int)y_w8a16[i]);
    printf("\n");

    weights_free(&weights);

    /* Q6.10 LUT 검증: LSB 단위 diff 2 이하, 0.5(512) 초과 픽셀 5% 미만 */
    if (max_diff <= 2.0f && (float)over_half / (float)out_elems < 0.05f) {
        printf("\nResult: OK (Conv+SiLU W8A16 Q6.10 matches quantized reference)\n");
        return 0;
    }
    printf("\nResult: NG (diff or outlier count too large)\n");
    return 1;
}
