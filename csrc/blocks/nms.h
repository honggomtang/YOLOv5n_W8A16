#ifndef NMS_H
#define NMS_H

#include <stdint.h>
#include "decode.h"  // detection_t 구조체 사용

// IoU 계산 (두 detection 간의 Intersection over Union)
// 반환값: IoU 값 (0.0 ~ 1.0)
float calculate_iou(const detection_t* box1, const detection_t* box2);

// Non-Maximum Suppression
// 입력: confidence로 정렬된 detection 배열 (내림차순)
// 출력: NMS 적용 후 남은 detection 배열 (동적 할당, 호출자가 해제해야 함)
// 반환값: 0 성공, -1 실패
int nms(
    detection_t* detections,           // 입력: detection 배열
    int32_t num_detections,            // 입력: detection 개수
    detection_t** output_detections,    // 출력: NMS 적용 후 detection 배열 (동적 할당)
    int32_t* output_count,             // 출력: 남은 detection 개수
    float iou_threshold,               // IoU 임계값 (일반적으로 0.45)
    int32_t max_detections);           // 최대 detection 개수

#endif // NMS_H
