#include <stdio.h>
#include <math.h>

#include "./test_vectors_detect.h"
#include "../csrc/utils/weights_loader.h"
#include "../csrc/blocks/detect.h"
#include "../csrc/blocks/conv.h"

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// 헬퍼 매크로: 텐서 이름으로 데이터 가져오기
#define W(name) weights_get_tensor_data(&weights, name)

int main(void) {
    // .bin 파일에서 가중치 로드
    weights_loader_t weights;
    if (weights_load_from_file("assets/weights.bin", &weights) != 0) {
        fprintf(stderr, "Failed to load weights.bin\n");
        return 1;
    }
    
    const int n = 1;

    // 출력 버퍼 (cv2, cv3 각각)
    static float p3_cv2_out[TV_DETECT_P3_CV2_C * TV_DETECT_P3_CV2_H * TV_DETECT_P3_CV2_W];
    static float p3_cv3_out[TV_DETECT_P3_CV3_C * TV_DETECT_P3_CV3_H * TV_DETECT_P3_CV3_W];
    static float p4_cv2_out[TV_DETECT_P4_CV2_C * TV_DETECT_P4_CV2_H * TV_DETECT_P4_CV2_W];
    static float p4_cv3_out[TV_DETECT_P4_CV3_C * TV_DETECT_P4_CV3_H * TV_DETECT_P4_CV3_W];
    static float p5_cv2_out[TV_DETECT_P5_CV2_C * TV_DETECT_P5_CV2_H * TV_DETECT_P5_CV2_W];
    static float p5_cv3_out[TV_DETECT_P5_CV3_C * TV_DETECT_P5_CV3_H * TV_DETECT_P5_CV3_W];

    int all_ok = 1;

    // Detect Head: P3, P4, P5 각각에 cv2 (bbox)와 cv3 (class) Sequential 적용
    detect_head_nchw_f32(
        // P3 입력 및 cv2, cv3 파라미터
        tv_detect_p3, TV_DETECT_P3_C, TV_DETECT_P3_H, TV_DETECT_P3_W,
        // cv2[0]
        W("model.24.cv2.0.0.conv.weight"), 64,
        W("model.24.cv2.0.0.bn.weight"), W("model.24.cv2.0.0.bn.bias"),
        W("model.24.cv2.0.0.bn.running_mean"), W("model.24.cv2.0.0.bn.running_var"),
        // cv2[1]
        W("model.24.cv2.0.1.conv.weight"),
        W("model.24.cv2.0.1.bn.weight"), W("model.24.cv2.0.1.bn.bias"),
        W("model.24.cv2.0.1.bn.running_mean"), W("model.24.cv2.0.1.bn.running_var"),
        // cv2[2]
        W("model.24.cv2.0.2.weight"), W("model.24.cv2.0.2.bias"),
        // cv3[0]
        W("model.24.cv3.0.0.conv.weight"), 80,
        W("model.24.cv3.0.0.bn.weight"), W("model.24.cv3.0.0.bn.bias"),
        W("model.24.cv3.0.0.bn.running_mean"), W("model.24.cv3.0.0.bn.running_var"),
        // cv3[1]
        W("model.24.cv3.0.1.conv.weight"),
        W("model.24.cv3.0.1.bn.weight"), W("model.24.cv3.0.1.bn.bias"),
        W("model.24.cv3.0.1.bn.running_mean"), W("model.24.cv3.0.1.bn.running_var"),
        // cv3[2]
        W("model.24.cv3.0.2.weight"), W("model.24.cv3.0.2.bias"),
        // P4 입력 및 cv2, cv3 파라미터
        tv_detect_p4, TV_DETECT_P4_C, TV_DETECT_P4_H, TV_DETECT_P4_W,
        // cv2[0]
        W("model.24.cv2.1.0.conv.weight"), 64,
        W("model.24.cv2.1.0.bn.weight"), W("model.24.cv2.1.0.bn.bias"),
        W("model.24.cv2.1.0.bn.running_mean"), W("model.24.cv2.1.0.bn.running_var"),
        // cv2[1]
        W("model.24.cv2.1.1.conv.weight"),
        W("model.24.cv2.1.1.bn.weight"), W("model.24.cv2.1.1.bn.bias"),
        W("model.24.cv2.1.1.bn.running_mean"), W("model.24.cv2.1.1.bn.running_var"),
        // cv2[2]
        W("model.24.cv2.1.2.weight"), W("model.24.cv2.1.2.bias"),
        // cv3[0]
        W("model.24.cv3.1.0.conv.weight"), 80,
        W("model.24.cv3.1.0.bn.weight"), W("model.24.cv3.1.0.bn.bias"),
        W("model.24.cv3.1.0.bn.running_mean"), W("model.24.cv3.1.0.bn.running_var"),
        // cv3[1]
        W("model.24.cv3.1.1.conv.weight"),
        W("model.24.cv3.1.1.bn.weight"), W("model.24.cv3.1.1.bn.bias"),
        W("model.24.cv3.1.1.bn.running_mean"), W("model.24.cv3.1.1.bn.running_var"),
        // cv3[2]
        W("model.24.cv3.1.2.weight"), W("model.24.cv3.1.2.bias"),
        // P5 입력 및 cv2, cv3 파라미터
        tv_detect_p5, TV_DETECT_P5_C, TV_DETECT_P5_H, TV_DETECT_P5_W,
        // cv2[0]
        W("model.24.cv2.2.0.conv.weight"), 64,
        W("model.24.cv2.2.0.bn.weight"), W("model.24.cv2.2.0.bn.bias"),
        W("model.24.cv2.2.0.bn.running_mean"), W("model.24.cv2.2.0.bn.running_var"),
        // cv2[1]
        W("model.24.cv2.2.1.conv.weight"),
        W("model.24.cv2.2.1.bn.weight"), W("model.24.cv2.2.1.bn.bias"),
        W("model.24.cv2.2.1.bn.running_mean"), W("model.24.cv2.2.1.bn.running_var"),
        // cv2[2]
        W("model.24.cv2.2.2.weight"), W("model.24.cv2.2.2.bias"),
        // cv3[0]
        W("model.24.cv3.2.0.conv.weight"), 80,
        W("model.24.cv3.2.0.bn.weight"), W("model.24.cv3.2.0.bn.bias"),
        W("model.24.cv3.2.0.bn.running_mean"), W("model.24.cv3.2.0.bn.running_var"),
        // cv3[1]
        W("model.24.cv3.2.1.conv.weight"),
        W("model.24.cv3.2.1.bn.weight"), W("model.24.cv3.2.1.bn.bias"),
        W("model.24.cv3.2.1.bn.running_mean"), W("model.24.cv3.2.1.bn.running_var"),
        // cv3[2]
        W("model.24.cv3.2.2.weight"), W("model.24.cv3.2.2.bias"),
        1e-3f,
        // 출력 (cv2, cv3 각각)
        p3_cv2_out, TV_DETECT_P3_CV2_H, TV_DETECT_P3_CV2_W,
        p3_cv3_out, TV_DETECT_P3_CV3_H, TV_DETECT_P3_CV3_W,
        p4_cv2_out, TV_DETECT_P4_CV2_H, TV_DETECT_P4_CV2_W,
        p4_cv3_out, TV_DETECT_P4_CV3_H, TV_DETECT_P4_CV3_W,
        p5_cv2_out, TV_DETECT_P5_CV2_H, TV_DETECT_P5_CV2_W,
        p5_cv3_out, TV_DETECT_P5_CV3_H, TV_DETECT_P5_CV3_W);

    // P3 cv2 검증
    {
        int elems = TV_DETECT_P3_CV2_C * TV_DETECT_P3_CV2_H * TV_DETECT_P3_CV2_W;
        float diff = max_abs_diff(p3_cv2_out, tv_detect_p3_cv2, elems);
        printf("P3 cv2 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // P3 cv3 검증
    {
        int elems = TV_DETECT_P3_CV3_C * TV_DETECT_P3_CV3_H * TV_DETECT_P3_CV3_W;
        float diff = max_abs_diff(p3_cv3_out, tv_detect_p3_cv3, elems);
        printf("P3 cv3 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // P4 cv2 검증
    {
        int elems = TV_DETECT_P4_CV2_C * TV_DETECT_P4_CV2_H * TV_DETECT_P4_CV2_W;
        float diff = max_abs_diff(p4_cv2_out, tv_detect_p4_cv2, elems);
        printf("P4 cv2 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // P4 cv3 검증
    {
        int elems = TV_DETECT_P4_CV3_C * TV_DETECT_P4_CV3_H * TV_DETECT_P4_CV3_W;
        float diff = max_abs_diff(p4_cv3_out, tv_detect_p4_cv3, elems);
        printf("P4 cv3 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // P5 cv2 검증
    {
        int elems = TV_DETECT_P5_CV2_C * TV_DETECT_P5_CV2_H * TV_DETECT_P5_CV2_W;
        float diff = max_abs_diff(p5_cv2_out, tv_detect_p5_cv2, elems);
        printf("P5 cv2 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    // P5 cv3 검증
    {
        int elems = TV_DETECT_P5_CV3_C * TV_DETECT_P5_CV3_H * TV_DETECT_P5_CV3_W;
        float diff = max_abs_diff(p5_cv3_out, tv_detect_p5_cv3, elems);
        printf("P5 cv3 diff = %g", diff);
        if (diff < 1e-4f) printf(" OK\n"); else { printf(" NG\n"); all_ok = 0; }
    }

    weights_free(&weights);
    
    if (all_ok) {
        printf("\nAll detections OK\n");
        return 0;
    }
    printf("\nSome detections failed\n");
    return 1;
}
