#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test_vectors_nms_full.h"
#include "../csrc/blocks/nms.h"

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

// 두 detection 배열 비교 (순서 무관)
static int compare_detections(
    const detection_t* a, int32_t num_a,
    const detection_t* b, int32_t num_b,
    float tolerance) {
    
    if (num_a != num_b) {
        printf("Count mismatch: %d vs %d\n", num_a, num_b);
        return 0;
    }
    
    // 각 detection을 찾아서 비교 (순서 무관)
    int* matched = (int*)calloc(num_b, sizeof(int));
    if (!matched) return 0;
    
    int all_matched = 1;
    for (int i = 0; i < num_a; i++) {
        int found = 0;
        for (int j = 0; j < num_b; j++) {
            if (matched[j]) continue;
            
            // 같은 클래스이고 bbox가 비슷한지 확인
            if (a[i].cls_id == b[j].cls_id) {
                float dx = fabsf(a[i].x - b[j].x);
                float dy = fabsf(a[i].y - b[j].y);
                float dw = fabsf(a[i].w - b[j].w);
                float dh = fabsf(a[i].h - b[j].h);
                float dconf = fabsf(a[i].conf - b[j].conf);
                
                if (dx < tolerance && dy < tolerance && 
                    dw < tolerance && dh < tolerance && 
                    dconf < tolerance) {
                    matched[j] = 1;
                    found = 1;
                    break;
                }
            }
        }
        if (!found) {
            printf("Detection [%d] not matched: cls=%d conf=%.4f bbox=(%.4f,%.4f,%.4f,%.4f)\n",
                   i, a[i].cls_id, a[i].conf, a[i].x, a[i].y, a[i].w, a[i].h);
            all_matched = 0;
        }
    }
    
    free(matched);
    return all_matched;
}

int main(void) {
    printf("=== NMS Full Pipeline Test ===\n\n");
    
    // Python 참조 결과
    const int32_t num_ref = TV_NMS_FULL_NUM_AFTER;
    printf("Python NMS result: %d detections\n", num_ref);
    
    // 테스트용 입력: Python 결과를 기반으로 중복 추가 (NMS 전 시뮬레이션)
    // 실제로는 decode 블록의 출력을 사용
    const int32_t num_input = num_ref * 2;  // 중복 추가
    detection_t* input_detections = (detection_t*)malloc(num_input * sizeof(detection_t));
    if (!input_detections) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Python 결과를 복사하고, 각 detection에 약간의 변형을 추가 (중복 시뮬레이션)
    for (int i = 0; i < num_ref; i++) {
        input_detections[i] = tv_nms_full_after[i];
    }
    
    // 중복 detection 추가 (약간의 offset으로 겹치게)
    for (int i = 0; i < num_ref; i++) {
        input_detections[num_ref + i] = tv_nms_full_after[i];
        input_detections[num_ref + i].x += 0.01f;  // 약간 이동 (겹침)
        input_detections[num_ref + i].y += 0.01f;
        input_detections[num_ref + i].conf *= 0.9f;  // 낮은 confidence
    }
    
    // confidence로 정렬 (NMS 전 필수)
    sort_detections_by_conf(input_detections, num_input);
    
    printf("Input detections (before NMS): %d\n", num_input);
    
    // NMS 적용
    detection_t* output_detections = NULL;
    int32_t output_count = 0;
    
    int ret = nms(
        input_detections, num_input,
        &output_detections, &output_count,
        TV_NMS_FULL_IOU_THRESHOLD,
        TV_NMS_FULL_MAX_DETECTIONS);
    
    if (ret != 0) {
        fprintf(stderr, "NMS failed\n");
        free(input_detections);
        return 1;
    }
    
    printf("C NMS result: %d detections\n", output_count);
    
    // Python 결과와 비교
    float tolerance = 0.01f;  // 1% 허용 오차
    int match = compare_detections(
        output_detections, output_count,
        tv_nms_full_after, num_ref,
        tolerance);
    
    if (match) {
        printf("\n✅ All detections matched (tolerance=%.2f)\n", tolerance);
    } else {
        printf("\n❌ Some detections did not match\n");
    }
    
    // 상세 비교
    printf("\n=== Detailed Comparison ===\n");
    printf("Python: %d detections\n", num_ref);
    printf("C NMS:  %d detections\n", output_count);
    
    if (output_count > 0 && num_ref > 0) {
        printf("\nFirst few detections:\n");
        int max_show = output_count < num_ref ? output_count : num_ref;
        max_show = max_show < 5 ? max_show : 5;
        
        for (int i = 0; i < max_show; i++) {
            printf("  [%d] Python: cls=%d conf=%.4f\n", i, 
                   tv_nms_full_after[i].cls_id, tv_nms_full_after[i].conf);
            if (i < output_count) {
                printf("       C NMS:  cls=%d conf=%.4f\n", 
                       output_detections[i].cls_id, output_detections[i].conf);
            }
        }
    }
    
    // 정리
    free(input_detections);
    if (output_detections) {
        free(output_detections);
    }
    
    return match ? 0 : 1;
}
