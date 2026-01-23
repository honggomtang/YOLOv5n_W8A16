#ifndef DETECT_H
#define DETECT_H

#include <stdint.h>

// Detect Head: P3, P4, P5 각각에 cv2 (bbox)와 cv3 (class) Sequential 적용
// YOLOv5nu Detect 구조:
// - cv2: bbox prediction Sequential (3개 레이어) -> 64채널 출력
//   [0]: Conv (입력->64, 3x3) + BN + SiLU
//   [1]: Conv (64->64, 3x3) + BN + SiLU
//   [2]: Conv2d (64->64, 1x1) + bias (BN/SiLU 없음)
// - cv3: class prediction Sequential (3개 레이어) -> 80채널 출력
//   [0]: Conv (입력->80, 3x3) + BN + SiLU
//   [1]: Conv (80->80, 3x3) + BN + SiLU
//   [2]: Conv2d (80->80, 1x1) + bias (BN/SiLU 없음)
//
// 입력: P3 (64채널), P4 (128채널), P5 (256채널)
// 출력: cv2는 각각 (64채널), cv3는 각각 (80채널)
void detect_head_nchw_f32(
    // P3 입력 및 cv2, cv3 파라미터 (cv2와 cv3 각각 3개 Conv)
    const float* p3, int32_t p3_c, int32_t p3_h, int32_t p3_w,
    const float* p3_cv2_0_w, int32_t p3_cv2_0_c_out, const float* p3_cv2_0_gamma, const float* p3_cv2_0_beta, const float* p3_cv2_0_mean, const float* p3_cv2_0_var,
    const float* p3_cv2_1_w, const float* p3_cv2_1_gamma, const float* p3_cv2_1_beta, const float* p3_cv2_1_mean, const float* p3_cv2_1_var,
    const float* p3_cv2_2_w, const float* p3_cv2_2_bias,
    const float* p3_cv3_0_w, int32_t p3_cv3_0_c_out, const float* p3_cv3_0_gamma, const float* p3_cv3_0_beta, const float* p3_cv3_0_mean, const float* p3_cv3_0_var,
    const float* p3_cv3_1_w, const float* p3_cv3_1_gamma, const float* p3_cv3_1_beta, const float* p3_cv3_1_mean, const float* p3_cv3_1_var,
    const float* p3_cv3_2_w, const float* p3_cv3_2_bias,
    // P4 입력 및 cv2, cv3 파라미터 (similar)
    const float* p4, int32_t p4_c, int32_t p4_h, int32_t p4_w,
    const float* p4_cv2_0_w, int32_t p4_cv2_0_c_out, const float* p4_cv2_0_gamma, const float* p4_cv2_0_beta, const float* p4_cv2_0_mean, const float* p4_cv2_0_var,
    const float* p4_cv2_1_w, const float* p4_cv2_1_gamma, const float* p4_cv2_1_beta, const float* p4_cv2_1_mean, const float* p4_cv2_1_var,
    const float* p4_cv2_2_w, const float* p4_cv2_2_bias,
    const float* p4_cv3_0_w, int32_t p4_cv3_0_c_out, const float* p4_cv3_0_gamma, const float* p4_cv3_0_beta, const float* p4_cv3_0_mean, const float* p4_cv3_0_var,
    const float* p4_cv3_1_w, const float* p4_cv3_1_gamma, const float* p4_cv3_1_beta, const float* p4_cv3_1_mean, const float* p4_cv3_1_var,
    const float* p4_cv3_2_w, const float* p4_cv3_2_bias,
    // P5 입력 및 cv2, cv3 파라미터 (similar)
    const float* p5, int32_t p5_c, int32_t p5_h, int32_t p5_w,
    const float* p5_cv2_0_w, int32_t p5_cv2_0_c_out, const float* p5_cv2_0_gamma, const float* p5_cv2_0_beta, const float* p5_cv2_0_mean, const float* p5_cv2_0_var,
    const float* p5_cv2_1_w, const float* p5_cv2_1_gamma, const float* p5_cv2_1_beta, const float* p5_cv2_1_mean, const float* p5_cv2_1_var,
    const float* p5_cv2_2_w, const float* p5_cv2_2_bias,
    const float* p5_cv3_0_w, int32_t p5_cv3_0_c_out, const float* p5_cv3_0_gamma, const float* p5_cv3_0_beta, const float* p5_cv3_0_mean, const float* p5_cv3_0_var,
    const float* p5_cv3_1_w, const float* p5_cv3_1_gamma, const float* p5_cv3_1_beta, const float* p5_cv3_1_mean, const float* p5_cv3_1_var,
    const float* p5_cv3_2_w, const float* p5_cv3_2_bias,
    float eps,
    // 출력 (cv2, cv3 각각)
    float* p3_cv2_out, int32_t p3_cv2_out_h, int32_t p3_cv2_out_w,
    float* p3_cv3_out, int32_t p3_cv3_out_h, int32_t p3_cv3_out_w,
    float* p4_cv2_out, int32_t p4_cv2_out_h, int32_t p4_cv2_out_w,
    float* p4_cv3_out, int32_t p4_cv3_out_h, int32_t p4_cv3_out_w,
    float* p5_cv2_out, int32_t p5_cv2_out_h, int32_t p5_cv2_out_w,
    float* p5_cv3_out, int32_t p5_cv3_out_h, int32_t p5_cv3_out_w);

#endif // DETECT_H
