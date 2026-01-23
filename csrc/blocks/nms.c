#include "nms.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// IoU 계산: 두 detection 간의 Intersection over Union
float calculate_iou(const detection_t* box1, const detection_t* box2) {
    if (!box1 || !box2) return 0.0f;
    
    // center+size 형식을 x1, y1, x2, y2로 변환
    float x1_1 = box1->x - box1->w / 2.0f;
    float y1_1 = box1->y - box1->h / 2.0f;
    float x2_1 = box1->x + box1->w / 2.0f;
    float y2_1 = box1->y + box1->h / 2.0f;
    
    float x1_2 = box2->x - box2->w / 2.0f;
    float y1_2 = box2->y - box2->h / 2.0f;
    float x2_2 = box2->x + box2->w / 2.0f;
    float y2_2 = box2->y + box2->h / 2.0f;
    
    // 교집합 계산
    float x1_i = fmaxf(x1_1, x1_2);
    float y1_i = fmaxf(y1_1, y1_2);
    float x2_i = fminf(x2_1, x2_2);
    float y2_i = fminf(y2_1, y2_2);
    
    if (x2_i < x1_i || y2_i < y1_i) {
        return 0.0f;  // 교집합 없음
    }
    
    float intersection = (x2_i - x1_i) * (y2_i - y1_i);
    
    // 합집합 계산
    float area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    float union_area = area1 + area2 - intersection;
    
    if (union_area <= 0.0f) return 0.0f;
    
    return intersection / union_area;
}

// Non-Maximum Suppression
// Greedy NMS 알고리즘: confidence가 높은 순서대로 처리하며, 같은 클래스의 겹치는 detection 제거
int nms(
    detection_t* detections,
    int32_t num_detections,
    detection_t** output_detections,
    int32_t* output_count,
    float iou_threshold,
    int32_t max_detections) {
    
    if (!detections || num_detections <= 0 || !output_detections || !output_count) {
        return -1;
    }
    
    *output_detections = NULL;
    *output_count = 0;
    
    // 주의: 입력 detection 배열은 이미 confidence 내림차순으로 정렬되어 있어야 함
    // Python의 torchvision.ops.nms와 동일한 동작:
    // 1. confidence로 정렬 (이미 완료)
    // 2. max_nms 개수만큼만 처리
    // 3. 같은 클래스의 겹치는 detection 제거
    
    // 유지할 detection 추적
    int* keep = (int*)calloc(num_detections, sizeof(int));
    if (!keep) return -1;
    
    int keep_count = 0;
    
    // Greedy NMS: 각 detection에 대해 겹치는 것들 제거
    for (int i = 0; i < num_detections && keep_count < max_detections; i++) {
        if (keep[i] == -1) continue;  // 이미 제거됨
        
        keep[i] = 1;  // 유지
        keep_count++;
        
        // 같은 클래스의 겹치는 detection 제거
        for (int j = i + 1; j < num_detections; j++) {
            if (keep[j] == -1) continue;  // 이미 제거됨
            
            // 같은 클래스인지 확인
            if (detections[i].cls_id != detections[j].cls_id) continue;
            
            float iou = calculate_iou(&detections[i], &detections[j]);
            if (iou > iou_threshold) {
                keep[j] = -1;  // 제거
            }
        }
    }
    
    // 출력 배열 할당
    if (keep_count > 0) {
        *output_detections = (detection_t*)malloc(keep_count * sizeof(detection_t));
        if (!*output_detections) {
            free(keep);
            return -1;
        }
        
        int idx = 0;
        for (int i = 0; i < num_detections; i++) {
            if (keep[i] == 1) {
                (*output_detections)[idx++] = detections[i];
            }
        }
    }
    
    *output_count = keep_count;
    free(keep);
    
    return 0;
}
