#include <stdio.h>
#include <math.h>

#include "test_vectors_upsample.h"

#include "../csrc/operations/upsample_w8a32.h"

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    const int n = TV_UPSAMPLE_X_N;
    const int c = TV_UPSAMPLE_X_C;
    const int h = TV_UPSAMPLE_X_H;
    const int w = TV_UPSAMPLE_X_W;

    const int h_out = TV_UPSAMPLE_Y_H;
    const int w_out = TV_UPSAMPLE_Y_W;

    static float y_out[TV_UPSAMPLE_X_N * TV_UPSAMPLE_Y_C * TV_UPSAMPLE_Y_H * TV_UPSAMPLE_Y_W];

    upsample_nearest2x_nchw_f32_w8a32(tv_upsample_x, n, c, h, w, y_out);

    const int elems = n * TV_UPSAMPLE_Y_C * h_out * w_out;
    float diff = max_abs_diff(y_out, tv_upsample_y, elems);
    printf("upsample max_abs_diff = %g\n", diff);

    if (diff < 1e-4f) {
        printf("OK\n");
        return 0;
    }
    printf("NG\n");
    return 1;
}
