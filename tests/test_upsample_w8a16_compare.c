/*
 * upsample nearest 2x W8A32 vs W8A16 비교
 * - 동일 입력 float → Q6.10으로 변환해 W8A16에 전달
 * - 스케일 변화 없음 → ref vs (float)y/1024
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "../csrc/operations/upsample_w8a32.h"
#include "../csrc/operations/upsample_w8a16.h"

#define Q6_10_SCALE 1024

static void float_to_q610(const float* x, int16_t* y, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i] * (float)Q6_10_SCALE;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        y[i] = (int16_t)(int32_t)v;
    }
}

int main(void) {
    printf("=== upsample_nearest2x W8A32 vs W8A16 compare ===\n\n");

    const int n = 1, c = 4, h = 4, w = 4;
    const int out_h = h * 2, out_w = w * 2;
    const int in_elems = n * c * h * w;
    const int out_elems = n * c * out_h * out_w;

    static float x_f[1 * 4 * 4 * 4];
    static int16_t x_q[1 * 4 * 4 * 4];
    static float y_f[1 * 4 * 8 * 8];
    static int16_t y_q[1 * 4 * 8 * 8];

    for (int i = 0; i < in_elems; i++)
        x_f[i] = (float)((i * 17 + 3) % 40) / 20.0f - 1.0f;
    float_to_q610(x_f, x_q, in_elems);

    upsample_nearest2x_nchw_f32_w8a32(x_f, n, c, h, w, y_f);
    upsample_nearest2x_nchw_w8a16(x_q, n, c, h, w, y_q);

    float max_diff = 0.0f;
    for (int i = 0; i < out_elems; i++) {
        float ref = y_f[i];
        float a16 = (float)y_q[i] / (float)Q6_10_SCALE;
        float d = fabsf(ref - a16);
        if (d > max_diff) max_diff = d;
    }

    printf("Shape: 1*%d*%d*%d -> nearest2x -> 1*%d*%d*%d\n", c, h, w, c, out_h, out_w);
    printf("Max |ref - W8A16/1024|: %g\n", (double)max_diff);
    printf("Sample ref[0..3]: %.4f %.4f %.4f %.4f\n", (double)y_f[0], (double)y_f[1], (double)y_f[2], (double)y_f[3]);
    printf("Sample W8A16[0..3]: %d %d %d %d\n", (int)y_q[0], (int)y_q[1], (int)y_q[2], (int)y_q[3]);

    if (max_diff < 1.0f / (float)Q6_10_SCALE * 2.0f) {
        printf("\nResult: OK (upsample_nearest2x_w8a16 matches W8A32)\n");
        return 0;
    }
    printf("\nResult: NG\n");
    return 1;
}
