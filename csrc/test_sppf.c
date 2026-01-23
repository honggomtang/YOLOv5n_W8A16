#include <stdio.h>
#include <math.h>

#include "../assets/weights.h"
#include "../tests/test_vectors_sppf.h"

#include "blocks/sppf.h"

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    const int n = TV_SPPF_X_N;
    const int c_in = TV_SPPF_X_C;
    const int h = TV_SPPF_X_H;
    const int w = TV_SPPF_X_W;

    const int c_out = TV_SPPF_Y_C;
    const int h_out = TV_SPPF_Y_H;
    const int w_out = TV_SPPF_Y_W;

    static float y_out[TV_SPPF_X_N * TV_SPPF_Y_C * TV_SPPF_Y_H * TV_SPPF_Y_W];

    // YOLOv5n SPPF(Layer 9): cv1(256->128) + pool(k=5)x3 + concat + cv2(512->256)
    sppf_nchw_f32(
        tv_sppf_x, n, c_in, h, w,
        // cv1
        model_9_cv1_conv_weight, 128,
        model_9_cv1_bn_weight, model_9_cv1_bn_bias,
        model_9_cv1_bn_running_mean, model_9_cv1_bn_running_var,
        // cv2
        model_9_cv2_conv_weight, 256,
        model_9_cv2_bn_weight, model_9_cv2_bn_bias,
        model_9_cv2_bn_running_mean, model_9_cv2_bn_running_var,
        // pool
        5,
        1e-3f,
        y_out);

    const int elems = n * c_out * h_out * w_out;
    float diff = max_abs_diff(y_out, tv_sppf_y, elems);
    printf("sppf max_abs_diff = %g\n", diff);

    if (diff < 1e-4f) {
        printf("OK\n");
        return 0;
    }
    printf("NG\n");
    return 1;
}
