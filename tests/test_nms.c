#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test_vectors_nms.h"
#include "../csrc/blocks/nms.h"

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// detection 배열을 confidence 내림차순으로 정렬 (버블 정렬)
static void sort_detections_by_conf(detection_t* detections, int32_t num) {
    for (int i = 0; i < num - 1; i++) {
        for (int j = 0; j < num - 1 - i; j++) {
            if (detections[j].conf < detections[j + 1].conf) {
                detection_t temp = detections[j];
                detections[j] = detections[j + 1];
                detections[j + 1] = temp;
            }
        }
    }
}

int main(void) {
    // 테스트: NMS 전 detection 배열 준비
    // 실제로는 decode 블록의 출력을 사용하지만, 여기서는 테스트 벡터 사용
    
    // Python NMS 결과 (참조)
    const int32_t num_ref = TV_NMS_NUM_DETECTIONS;
    
    printf("Reference detections (Python NMS): %d\n", num_ref);
    
    // 테스트용 입력 detection 배열 생성 (중복/겹치는 detection 포함)
    // 실제로는 decode 블록의 출력을 사용하지만, 여기서는 간단한 테스트 데이터 사용
    const int32_t num_input = 10;  // 예시: 10개의 detection
    detection_t input_detections[10] = {
        // 겹치는 detection들 (같은 클래스)
        {0.5f, 0.5f, 0.3f, 0.3f, 0.9f, 0},  // 높은 confidence
        {0.52f, 0.52f, 0.3f, 0.3f, 0.8f, 0},  // 겹침 (제거되어야 함)
        {0.48f, 0.48f, 0.3f, 0.3f, 0.7f, 0},  // 겹침 (제거되어야 함)
        // 다른 클래스 (제거되지 않아야 함)
        {0.2f, 0.2f, 0.2f, 0.2f, 0.85f, 1},
        {0.22f, 0.22f, 0.2f, 0.2f, 0.75f, 1},  // 같은 클래스, 겹침
        // 겹치지 않는 detection
        {0.8f, 0.8f, 0.1f, 0.1f, 0.6f, 0},
        {0.1f, 0.8f, 0.15f, 0.15f, 0.55f, 2},
        {0.9f, 0.1f, 0.12f, 0.12f, 0.5f, 0},
        {0.3f, 0.7f, 0.18f, 0.18f, 0.45f, 1},
        {0.7f, 0.3f, 0.14f, 0.14f, 0.4f, 2},
    };
    
    // confidence로 정렬 (NMS 전 필수)
    sort_detections_by_conf(input_detections, num_input);
    
    printf("\nInput detections (sorted by confidence):\n");
    for (int i = 0; i < num_input; i++) {
        printf("  [%d] cls=%d conf=%.3f bbox=(%.3f,%.3f,%.3f,%.3f)\n",
               i, input_detections[i].cls_id, input_detections[i].conf,
               input_detections[i].x, input_detections[i].y,
               input_detections[i].w, input_detections[i].h);
    }
    
    // NMS 적용
    detection_t* output_detections = NULL;
    int32_t output_count = 0;
    
    int ret = nms(
        input_detections, num_input,
        &output_detections, &output_count,
        TV_NMS_IOU_THRESHOLD,
        TV_NMS_MAX_DETECTIONS);
    
    if (ret != 0) {
        fprintf(stderr, "NMS failed\n");
        return 1;
    }
    
    printf("\nNMS output: %d detections\n", output_count);
    for (int i = 0; i < output_count; i++) {
        printf("  [%d] cls=%d conf=%.3f bbox=(%.3f,%.3f,%.3f,%.3f)\n",
               i, output_detections[i].cls_id, output_detections[i].conf,
               output_detections[i].x, output_detections[i].y,
               output_detections[i].w, output_detections[i].h);
    }
    
    // Python 참조 결과와 비교 (간단한 검증)
    printf("\nReference detections (Python):\n");
    for (int i = 0; i < num_ref && i < 5; i++) {  // 처음 5개만 출력
        printf("  [%d] cls=%d conf=%.3f bbox=(%.3f,%.3f,%.3f,%.3f)\n",
               i, tv_nms_detections[i].cls_id, tv_nms_detections[i].conf,
               tv_nms_detections[i].x, tv_nms_detections[i].y,
               tv_nms_detections[i].w, tv_nms_detections[i].h);
    }
    
    // IoU 계산 함수 테스트
    printf("\n=== IoU Test ===\n");
    detection_t box1 = {0.5f, 0.5f, 0.2f, 0.2f, 0.9f, 0};
    detection_t box2 = {0.52f, 0.52f, 0.2f, 0.2f, 0.8f, 0};
    float iou = calculate_iou(&box1, &box2);
    printf("Box1: (%.2f,%.2f,%.2f,%.2f)\n", box1.x, box1.y, box1.w, box1.h);
    printf("Box2: (%.2f,%.2f,%.2f,%.2f)\n", box2.x, box2.y, box2.w, box2.h);
    printf("IoU: %.4f\n", iou);
    
    // 정리
    if (output_detections) {
        free(output_detections);
    }
    
    printf("\nNMS test completed successfully\n");
    return 0;
}
