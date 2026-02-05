/*
 * concat W8A32 vs W8A16 비교
 * - 두 피처맵을 동일 Q6.10으로 만들어 concat. 스케일 일치 시 ref와 W8A16/1024 일치해야 함.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

#include "../csrc/operations/concat_w8a32.h"
#include "../csrc/operations/concat_w8a16.h"

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
    printf("=== concat W8A32 vs W8A16 compare (same Q6.10 scale) ===\n\n");

    const int n = 1, h = 4, w = 4, c1 = 2, c2 = 3;
    const int out_c = c1 + c2;
    const int in1_elems = n * c1 * h * w, in2_elems = n * c2 * h * w, out_elems = n * out_c * h * w;

    static float x1_f[1 * 2 * 4 * 4], x2_f[1 * 3 * 4 * 4];
    static int16_t x1_q[1 * 2 * 4 * 4], x2_q[1 * 3 * 4 * 4];
    static float y_f[1 * 5 * 4 * 4];
    static int16_t y_q[1 * 5 * 4 * 4];

    for (int i = 0; i < in1_elems; i++) x1_f[i] = (float)(i % 20) / 10.0f - 1.0f;
    for (int i = 0; i < in2_elems; i++) x2_f[i] = (float)((i * 7) % 20) / 10.0f - 1.0f;
    float_to_q610(x1_f, x1_q, in1_elems);
    float_to_q610(x2_f, x2_q, in2_elems);

    concat_nchw_f32_w8a32(x1_f, c1, x2_f, c2, n, h, w, y_f);
    concat_nchw_w8a16(x1_q, c1, x2_q, c2, n, h, w, y_q);

    float max_diff = 0.0f;
    for (int i = 0; i < out_elems; i++) {
        float ref = y_f[i];
        float a16 = (float)y_q[i] / (float)Q6_10_SCALE;
        float d = fabsf(ref - a16);
        if (d > max_diff) max_diff = d;
    }

    printf("Shape: [1*%d*%d*%d] + [1*%d*%d*%d] -> 1*%d*%d*%d\n", c1, h, w, c2, h, w, out_c, h, w);
    printf("Max |ref - W8A16/1024|: %g (same scale → 0에 가까움)\n", (double)max_diff);
    printf("Sample ref[0..3]: %.4f %.4f %.4f %.4f\n", (double)y_f[0], (double)y_f[1], (double)y_f[2], (double)y_f[3]);
    printf("Sample W8A16[0..3]: %d %d %d %d\n", (int)y_q[0], (int)y_q[1], (int)y_q[2], (int)y_q[3]);

    if (max_diff < 1.0f / (float)Q6_10_SCALE * 2.0f) {
        printf("\nResult: OK (concat_w8a16 matches W8A32 when inputs same Q6.10)\n");
        return 0;
    }
    printf("\nResult: NG\n");
    return 1;
}
