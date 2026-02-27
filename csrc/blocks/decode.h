#ifndef DECODE_H
#define DECODE_H

#include <stdint.h>

typedef struct {
    float x, y, w, h;   // 중심 좌표 및 크기 (normalized)
    float conf;         // confidence score
    int32_t cls_id;     // class ID
} detection_t;

#if defined(__GNUC__)
#define YOLO_PACKED __attribute__((packed))
#else
#define YOLO_PACKED
#endif

typedef struct YOLO_PACKED {
    uint16_t x, y, w, h;   // 픽셀 좌표 (정수, 0~65535)
    uint8_t  class_id;     // 클래스 ID (0~79)
    uint8_t  confidence;   // 신뢰도 (0~255, conf*255)
    uint8_t  reserved[2];  // 8바이트 정렬용
} hw_detection_t;

int32_t decode_nchw_f32(
    const float* p3, int32_t p3_h, int32_t p3_w,
    const float* p4, int32_t p4_h, int32_t p4_w,
    const float* p5, int32_t p5_h, int32_t p5_w,
    int32_t num_classes,
    float conf_threshold,
    int32_t input_size,
    const float strides[3],
    const float anchors[3][6],
    detection_t* detections,
    int32_t max_detections);

#endif /* DECODE_H */
