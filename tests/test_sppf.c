/* SPPF 블록 테스트 (Layer 9: 256x20x20 → 256x20x20). */
#include <stdio.h>
#include <math.h>

#include "test_vectors_sppf.h"
#include "../csrc/utils/weights_loader.h"
#include "../csrc/blocks/sppf_w8a32.h"

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

#define W(name) weights_get_tensor_data(&weights, name)

int main(void) {
    printf("=== SPPF Block Test (Fused) ===\n\n");
    
    // 가중치 로드
    weights_loader_t weights;
    if (weights_load_from_file("assets/weights.bin", &weights) != 0) {
        fprintf(stderr, "Failed to load weights.bin\n");
        return 1;
    }
    
    // Layer 9 SPPF 가중치 (Fused: conv.weight + conv.bias)
    const float* cv1_w = W("model.9.cv1.conv.weight");
    const float* cv1_b = W("model.9.cv1.conv.bias");
    const float* cv2_w = W("model.9.cv2.conv.weight");
    const float* cv2_b = W("model.9.cv2.conv.bias");
    
    if (!cv1_w || !cv1_b || !cv2_w || !cv2_b) {
        fprintf(stderr, "Failed to find required tensors\n");
        weights_free(&weights);
        return 1;
    }
    
    const int n = TV_SPPF_X_N;
    const int c_in = TV_SPPF_X_C;
    const int h = TV_SPPF_X_H;
    const int w = TV_SPPF_X_W;
    const int c_out = TV_SPPF_Y_C;
    const int h_out = TV_SPPF_Y_H;
    const int w_out = TV_SPPF_Y_W;

    static float y_out[1 * 256 * 20 * 20];

    // SPPF 블록 실행 (Fused, FP32 가중치: scale=1, is_int8=0)
    sppf_nchw_f32_w8a32(
        tv_sppf_x, n, c_in, h, w,
        cv1_w, 1.0f, 0, 128, cv1_b,   // cv1: 256->128
        cv2_w, 1.0f, 0, 256, cv2_b,   // cv2: 512->256
        5,                             // pool_k=5
        y_out);

    const int elems = n * c_out * h_out * w_out;
    float diff = max_abs_diff(y_out, tv_sppf_y, elems);
    
    printf("Layer 9 (SPPF, k=5)\n");
    printf("  Input:  %d x %d x %d x %d\n", n, c_in, h, w);
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
