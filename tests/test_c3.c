/* C3 블록 테스트 (Layer 2: 32x160x160 → 32x160x160). */
#include <stdio.h>
#include <math.h>

#include "test_vectors_c3.h"
#include "../csrc/utils/weights_loader.h"
#include "../csrc/blocks/c3_w8a32.h"

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
    printf("=== C3 Block Test (Fused) ===\n\n");
    
    // 가중치 로드
    weights_loader_t weights;
    if (weights_load_from_file("assets/weights.bin", &weights) != 0) {
        fprintf(stderr, "Failed to load weights.bin\n");
        return 1;
    }
    
    // Layer 2 C3 가중치 (Fused: conv.weight + conv.bias)
    const float* cv1_w = W("model.2.cv1.conv.weight");
    const float* cv1_b = W("model.2.cv1.conv.bias");
    const float* cv2_w = W("model.2.cv2.conv.weight");
    const float* cv2_b = W("model.2.cv2.conv.bias");
    const float* cv3_w = W("model.2.cv3.conv.weight");
    const float* cv3_b = W("model.2.cv3.conv.bias");
    const float* bn_cv1_w = W("model.2.m.0.cv1.conv.weight");
    const float* bn_cv1_b = W("model.2.m.0.cv1.conv.bias");
    const float* bn_cv2_w = W("model.2.m.0.cv2.conv.weight");
    const float* bn_cv2_b = W("model.2.m.0.cv2.conv.bias");
    
    if (!cv1_w || !cv1_b || !cv2_w || !cv2_b || !cv3_w || !cv3_b ||
        !bn_cv1_w || !bn_cv1_b || !bn_cv2_w || !bn_cv2_b) {
        fprintf(stderr, "Failed to find required tensors\n");
        weights_free(&weights);
        return 1;
    }
    
    // YOLOv5n Layer 2 (C3): 입력 32x160x160 → 출력 32x160x160
    const int n = TV_C3_X_N;
    const int c_in = TV_C3_X_C;
    const int h = TV_C3_X_H;
    const int w = TV_C3_X_W;
    const int c_out = TV_C3_Y_C;
    
    static float y_out[1 * 32 * 160 * 160];
    
    // C3 블록 실행 (Fused). FP32 가중치이므로 scale=0, is_int8=0
    const void* bn_cv1_w_arr[1] = {bn_cv1_w};
    const void* bn_cv2_w_arr[1] = {bn_cv2_w};
    float bn_cv1_scale[1] = {0.f};
    int bn_cv1_is_int8[1] = {0};
    float bn_cv2_scale[1] = {0.f};
    int bn_cv2_is_int8[1] = {0};
    const float* bn_cv1_b_arr[1] = {bn_cv1_b};
    const float* bn_cv2_b_arr[1] = {bn_cv2_b};
    
    c3_nchw_f32_w8a32(
        tv_c3_x, n, c_in, h, w,
        (const void*)cv1_w, 0.f, 0, 16, cv1_b,
        (const void*)cv2_w, 0.f, 0, 16, cv2_b,
        (const void*)cv3_w, 0.f, 0, 32, cv3_b,
        1,
        bn_cv1_w_arr, bn_cv1_scale, bn_cv1_is_int8, bn_cv1_b_arr,
        bn_cv2_w_arr, bn_cv2_scale, bn_cv2_is_int8, bn_cv2_b_arr,
        1,
        y_out);
    
    const int elems = n * c_out * h * w;
    float diff = max_abs_diff(y_out, tv_c3_y, elems);
    
    printf("Layer 2 (C3, n=1)\n");
    printf("  Input:  %d x %d x %d x %d\n", n, c_in, h, w);
    printf("  Output: %d x %d x %d x %d\n", n, c_out, h, w);
    printf("  Max diff: %g\n\n", diff);
    
    weights_free(&weights);
    
    if (diff < 1e-4f) {
        printf("Result: OK\n");
        return 0;
    }
    printf("Result: NG\n");
    return 1;
}
