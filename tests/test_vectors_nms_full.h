#ifndef TEST_VECTORS_NMS_FULL_H
#define TEST_VECTORS_NMS_FULL_H

// 자동 생성됨 (NMS 전체 파이프라인 검증)
// Python YOLOv5n의 decode + NMS 결과

#include "../csrc/blocks/nms.h"

#define TV_NMS_FULL_NUM_BEFORE 4
#define TV_NMS_FULL_NUM_AFTER 4
#define TV_NMS_FULL_IOU_THRESHOLD 0.45f
#define TV_NMS_FULL_MAX_DETECTIONS 300
#define TV_NMS_FULL_IMG_W 1280
#define TV_NMS_FULL_IMG_H 720

// Python NMS 결과 (confidence 내림차순 정렬됨)
static const detection_t tv_nms_full_after[] = {
    { 3.48543227e-01f, 6.34308100e-01f, 4.60482836e-01f, 7.09898651e-01f, 9.24510837e-01f, 0.0 },
    { 7.39643455e-01f, 5.21223009e-01f, 3.09468567e-01f, 9.18127775e-01f, 8.97088289e-01f, 0.0 },
    { 7.93174088e-01f, 5.03688574e-01f, 3.51663604e-02f, 1.50776163e-01f, 3.63721311e-01f, 27.0 },
    { 8.47954154e-01f, 7.94745743e-01f, 8.16509277e-02f, 3.77022117e-01f, 2.76695222e-01f, 38.0 },
};

#endif // TEST_VECTORS_NMS_FULL_H
