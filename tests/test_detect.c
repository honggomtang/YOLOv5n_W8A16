/* Detect Head 테스트 (Layer 24: P3/P4/P5 → 255ch). */
#include <stdio.h>
#include <math.h>

#include "test_vectors_detect.h"
#include "../csrc/utils/weights_loader.h"
#include "../csrc/blocks/detect.h"

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
    printf("=== Detect Head Test (Anchor-based Fused) ===\n\n");
    
    // 가중치 로드
    weights_loader_t weights;
    if (weights_load_from_file("assets/weights.bin", &weights) != 0) {
        fprintf(stderr, "Failed to load weights.bin\n");
        return 1;
    }
    
    // Detect Head 가중치 (model.24.m.0/1/2)
    const float* m0_w = W("model.24.m.0.weight");
    const float* m0_b = W("model.24.m.0.bias");
    const float* m1_w = W("model.24.m.1.weight");
    const float* m1_b = W("model.24.m.1.bias");
    const float* m2_w = W("model.24.m.2.weight");
    const float* m2_b = W("model.24.m.2.bias");
    
    if (!m0_w || !m0_b || !m1_w || !m1_b || !m2_w || !m2_b) {
        fprintf(stderr, "Failed to find required tensors (model.24.m.*)\n");
        fprintf(stderr, "Make sure weights.bin is exported with --classic option\n");
        weights_free(&weights);
        return 1;
    }
    
    // 출력 버퍼
    static float p3_out[255 * 80 * 80];
    static float p4_out[255 * 40 * 40];
    static float p5_out[255 * 20 * 20];
    
    // Detect Head 실행 (FP32 가중치: scale=0, is_int8=0)
    detect_nchw_f32(
        tv_detect_p3, TV_DETECT_P3_C, TV_DETECT_P3_H, TV_DETECT_P3_W,
        tv_detect_p4, TV_DETECT_P4_C, TV_DETECT_P4_H, TV_DETECT_P4_W,
        tv_detect_p5, TV_DETECT_P5_C, TV_DETECT_P5_H, TV_DETECT_P5_W,
        (const void*)m0_w, 0.f, 0, m0_b,
        (const void*)m1_w, 0.f, 0, m1_b,
        (const void*)m2_w, 0.f, 0, m2_b,
        p3_out, p4_out, p5_out);
    
    int all_ok = 1;
    
    // P3 검증
    {
        int elems = 255 * TV_DETECT_P3_H * TV_DETECT_P3_W;
        float diff = max_abs_diff(p3_out, tv_detect_p3_out, elems);
        printf("P3 (80x80): max diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }
    
    // P4 검증
    {
        int elems = 255 * TV_DETECT_P4_H * TV_DETECT_P4_W;
        float diff = max_abs_diff(p4_out, tv_detect_p4_out, elems);
        printf("P4 (40x40): max diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }
    
    // P5 검증
    {
        int elems = 255 * TV_DETECT_P5_H * TV_DETECT_P5_W;
        float diff = max_abs_diff(p5_out, tv_detect_p5_out, elems);
        printf("P5 (20x20): max diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }
    
    weights_free(&weights);
    
    printf("\n");
    if (all_ok) {
        printf("Result: OK\n");
        return 0;
    }
    printf("Result: NG\n");
    return 1;
}
