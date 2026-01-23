#ifndef DECODE_H
#define DECODE_H

#include <stdint.h>

// Detection 결과 구조체
typedef struct {
    float x, y, w, h;  // bbox (normalized 0-1)
    float conf;         // confidence
    int32_t cls_id;     // class ID
} detection_t;

// Decode: cv2 (bbox)와 cv3 (class) 출력을 받아서 detection 결과로 변환
// YOLOv5nu (Universal) decode 로직 (DFL 구조):
// - 각 scale (P3, P4, P5)에 대해
// - 각 grid cell (y, x)에 대해
// - bbox decode (DFL), confidence 계산, threshold 적용
//
// cv2 출력 구조 (DFL):
// - (1, 64, H, W) = 4개 좌표(Left, Top, Right, Bottom) × 16채널(reg_max=16)
// - Channel 0~15: Left (L)
// - Channel 16~31: Top (T)
// - Channel 32~47: Right (R)
// - Channel 48~63: Bottom (B)
//
// cv3 출력:
// - (1, 80, H, W) - class prediction
//
// 출력: detection_t 배열 (최대 max_detections개)
int32_t decode_detections_nchw_f32(
    // P3 cv2, cv3 출력
    const float* p3_cv2, int32_t p3_h, int32_t p3_w,
    const float* p3_cv3, int32_t p3_cv3_c,
    // P4 cv2, cv3 출력
    const float* p4_cv2, int32_t p4_h, int32_t p4_w,
    const float* p4_cv3, int32_t p4_cv3_c,
    // P5 cv2, cv3 출력
    const float* p5_cv2, int32_t p5_h, int32_t p5_w,
    const float* p5_cv3, int32_t p5_cv3_c,
    // 파라미터
    int32_t num_classes,
    float conf_threshold,
    int32_t input_size,  // 원본 이미지 크기 (stride 계산용)
    const float strides[3],      // 각 scale의 stride
    // 출력
    detection_t* detections,
    int32_t max_detections);

#endif // DECODE_H
