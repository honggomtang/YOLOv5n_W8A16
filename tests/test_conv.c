/* Conv 블록 테스트 (Layer 0: 3x640x640 → 16x320x320). */
#include <stdio.h>
#include <math.h>

#include "test_vectors_conv.h"
#include "../csrc/utils/weights_loader.h"
#include "../csrc/blocks/conv_w8a32.h"

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(void) {
    printf("=== Conv Block Test (Fused) ===\n\n");
    
    weights_loader_t weights;
    if (weights_load_from_file("assets/weights.bin", &weights) != 0) {
        fprintf(stderr, "Failed to load weights.bin\n");
        return 1;
    }
    
    const float* conv_weight = weights_get_tensor_data(&weights, "model.0.conv.weight");
    const float* conv_bias = weights_get_tensor_data(&weights, "model.0.conv.bias");
    
    if (!conv_weight || !conv_bias) {
        fprintf(stderr, "Failed to find required tensors\n");
        weights_free(&weights);
        return 1;
    }
    
    // YOLOv5n conv0: Conv2d(3->16, k=6, s=2, p=2) + SiLU
    const int n = TV_X_N;
    const int c_in = TV_X_C;
    const int h_in = TV_X_H;
    const int w_in = TV_X_W;
    const int c_out = 16;
    const int k = 6;
    const int stride = 2;
    const int pad = 2;
    const int h_out = TV_Y_H;
    const int w_out = TV_Y_W;

    static float y_out[1 * 16 * 320 * 320];

    // Conv 블록 실행 (Fused: Conv + Bias + SiLU)
    conv_block_nchw_f32_w8a32(
        tv_x, n, c_in, h_in, w_in,
        conv_weight, c_out, k, k,
        stride, stride,
        pad, pad,
        conv_bias,
        y_out, h_out, w_out);

    const int elems = n * c_out * h_out * w_out;
    float diff = max_abs_diff(y_out, tv_y, elems);
    
    printf("Layer 0 (Conv 6x6 s2)\n");
    printf("  Input:  %d x %d x %d x %d\n", n, c_in, h_in, w_in);
    printf("  Output: %d x %d x %d x %d\n", n, c_out, h_out, w_out);
    printf("  Max diff: %g\n\n", diff);

    weights_free(&weights);
    
    if (diff < 1e-4f) {
        printf("Result: OK\n");
        return 0;
    }
    printf("Result: NG\n");
    return 1;
}
