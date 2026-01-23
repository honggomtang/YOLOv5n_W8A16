#include <stdio.h>
#include <math.h>
#include <string.h>

#include "./test_vectors_detect.h"
#include "./test_vectors_decode.h"

#include "../csrc/blocks/decode.h"

// Detection 비교 함수
static float detection_diff(const detection_t* a, const detection_t* b) {
    float diff = 0.0f;
    diff = fmaxf(diff, fabsf(a->x - b->x));
    diff = fmaxf(diff, fabsf(a->y - b->y));
    diff = fmaxf(diff, fabsf(a->w - b->w));
    diff = fmaxf(diff, fabsf(a->h - b->h));
    diff = fmaxf(diff, fabsf(a->conf - b->conf));
    if (a->cls_id != b->cls_id) {
        diff = fmaxf(diff, 1.0f);  // 클래스가 다르면 큰 차이로 표시
    }
    return diff;
}

int main(void) {
    // Decode 출력 버퍼
    static detection_t detections[100];
    
    // Strides
    const float strides[3] = {
        TV_DECODE_STRIDE_0,
        TV_DECODE_STRIDE_1,
        TV_DECODE_STRIDE_2
    };
    
    // Decode 실행
    int32_t num_detections = decode_detections_nchw_f32(
        // P3 cv2, cv3 출력
        tv_detect_p3_cv2, TV_DECODE_P3_CV2_H, TV_DECODE_P3_CV2_W,
        tv_detect_p3_cv3, TV_DECODE_P3_CV3_C,
        // P4 cv2, cv3 출력
        tv_detect_p4_cv2, TV_DECODE_P4_CV2_H, TV_DECODE_P4_CV2_W,
        tv_detect_p4_cv3, TV_DECODE_P4_CV3_C,
        // P5 cv2, cv3 출력
        tv_detect_p5_cv2, TV_DECODE_P5_CV2_H, TV_DECODE_P5_CV2_W,
        tv_detect_p5_cv3, TV_DECODE_P5_CV3_C,
        // 파라미터
        TV_DECODE_NUM_CLASSES,
        TV_DECODE_CONF_THRESHOLD,
        TV_DECODE_INPUT_SIZE,
        strides,
        // 출력
        detections,
        100);
    
    printf("C decoded %d detections\n", num_detections);
    printf("Python decoded %d detections\n", TV_DECODE_NUM_DETECTIONS);
    
    // 개수 비교
    if (num_detections != TV_DECODE_NUM_DETECTIONS) {
        printf("ERROR: Detection count mismatch: C=%d, Python=%d\n", 
               num_detections, TV_DECODE_NUM_DETECTIONS);
        return 1;
    }
    
    // C detection을 confidence 순으로 정렬 (Python과 동일하게)
    for (int i = 0; i < num_detections - 1; i++) {
        for (int j = i + 1; j < num_detections; j++) {
            if (detections[i].conf < detections[j].conf) {
                detection_t tmp = detections[i];
                detections[i] = detections[j];
                detections[j] = tmp;
            }
        }
    }
    
    // 각 detection 비교
    int all_ok = 1;
    float max_diff = 0.0f;
    
    for (int i = 0; i < num_detections; i++) {
        float diff = detection_diff(&detections[i], &tv_decode_detections[i]);
        max_diff = fmaxf(max_diff, diff);
        
        if (diff > 1e-2f) {  // 0.01 이상 차이면 오류 (약간의 수치 오차 허용)
            printf("Detection[%d] diff = %g", i, diff);
            printf("  C: x=%.6f y=%.6f w=%.6f h=%.6f conf=%.6f cls=%d\n",
                   detections[i].x, detections[i].y, detections[i].w,
                   detections[i].h, detections[i].conf, detections[i].cls_id);
            printf("  P: x=%.6f y=%.6f w=%.6f h=%.6f conf=%.6f cls=%d\n",
                   tv_decode_detections[i].x, tv_decode_detections[i].y,
                   tv_decode_detections[i].w, tv_decode_detections[i].h,
                   tv_decode_detections[i].conf, tv_decode_detections[i].cls_id);
            all_ok = 0;
        }
    }
    
    printf("\nMax detection diff = %g", max_diff);
    if (max_diff < 1e-2f) {
        printf(" OK\n");
    } else {
        printf(" NG\n");
        all_ok = 0;
    }
    
    if (all_ok) {
        printf("\nAll detections OK\n");
        return 0;
    }
    printf("\nSome detections failed\n");
    return 1;
}
