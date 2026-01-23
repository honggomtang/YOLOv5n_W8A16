#include "detect.h"
#include "conv.h"
#include "../operations/conv2d.h"

// Detect Head: P3, P4, P5 각각에 cv2 (bbox)와 cv3 (class) Sequential 적용
void detect_head_nchw_f32(
    // P3 입력 및 cv2, cv3 파라미터
    const float* p3, int32_t p3_c, int32_t p3_h, int32_t p3_w,
    const float* p3_cv2_0_w, int32_t p3_cv2_0_c_out, const float* p3_cv2_0_gamma, const float* p3_cv2_0_beta, const float* p3_cv2_0_mean, const float* p3_cv2_0_var,
    const float* p3_cv2_1_w, const float* p3_cv2_1_gamma, const float* p3_cv2_1_beta, const float* p3_cv2_1_mean, const float* p3_cv2_1_var,
    const float* p3_cv2_2_w, const float* p3_cv2_2_bias,
    const float* p3_cv3_0_w, int32_t p3_cv3_0_c_out, const float* p3_cv3_0_gamma, const float* p3_cv3_0_beta, const float* p3_cv3_0_mean, const float* p3_cv3_0_var,
    const float* p3_cv3_1_w, const float* p3_cv3_1_gamma, const float* p3_cv3_1_beta, const float* p3_cv3_1_mean, const float* p3_cv3_1_var,
    const float* p3_cv3_2_w, const float* p3_cv3_2_bias,
    // P4 입력 및 cv2, cv3 파라미터
    const float* p4, int32_t p4_c, int32_t p4_h, int32_t p4_w,
    const float* p4_cv2_0_w, int32_t p4_cv2_0_c_out, const float* p4_cv2_0_gamma, const float* p4_cv2_0_beta, const float* p4_cv2_0_mean, const float* p4_cv2_0_var,
    const float* p4_cv2_1_w, const float* p4_cv2_1_gamma, const float* p4_cv2_1_beta, const float* p4_cv2_1_mean, const float* p4_cv2_1_var,
    const float* p4_cv2_2_w, const float* p4_cv2_2_bias,
    const float* p4_cv3_0_w, int32_t p4_cv3_0_c_out, const float* p4_cv3_0_gamma, const float* p4_cv3_0_beta, const float* p4_cv3_0_mean, const float* p4_cv3_0_var,
    const float* p4_cv3_1_w, const float* p4_cv3_1_gamma, const float* p4_cv3_1_beta, const float* p4_cv3_1_mean, const float* p4_cv3_1_var,
    const float* p4_cv3_2_w, const float* p4_cv3_2_bias,
    // P5 입력 및 cv2, cv3 파라미터
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
    float* p5_cv3_out, int32_t p5_cv3_out_h, int32_t p5_cv3_out_w)
{
    static float p3_cv2_step0[1024 * 1024];
    static float p3_cv2_step1[1024 * 1024];
    static float p3_cv3_step0[1024 * 1024];
    static float p3_cv3_step1[1024 * 1024];
    static float p4_cv2_step0[1024 * 1024];
    static float p4_cv2_step1[1024 * 1024];
    static float p4_cv3_step0[1024 * 1024];
    static float p4_cv3_step1[1024 * 1024];
    static float p5_cv2_step0[1024 * 1024];
    static float p5_cv2_step1[1024 * 1024];
    static float p5_cv3_step0[1024 * 1024];
    static float p5_cv3_step1[1024 * 1024];
    
    // P3: cv2 Sequential (bbox prediction)
    // [0]: Conv (64->64, 3x3) + BN + SiLU
    conv_block_nchw_f32(
        p3, 1, p3_c, p3_h, p3_w,
        p3_cv2_0_w, p3_cv2_0_c_out, 3, 3,
        1, 1, 1, 1,
        p3_cv2_0_gamma, p3_cv2_0_beta,
        p3_cv2_0_mean, p3_cv2_0_var,
        eps,
        p3_cv2_step0, p3_h, p3_w);
    
    // [1]: Conv (64->64, 3x3) + BN + SiLU
    conv_block_nchw_f32(
        p3_cv2_step0, 1, 64, p3_h, p3_w,
        p3_cv2_1_w, 64, 3, 3,
        1, 1, 1, 1,
        p3_cv2_1_gamma, p3_cv2_1_beta,
        p3_cv2_1_mean, p3_cv2_1_var,
        eps,
        p3_cv2_step1, p3_h, p3_w);
    
    // [2]: Conv2d (64->64, 1x1) + bias (BN/SiLU 없음)
    conv2d_nchw_f32(
        p3_cv2_step1, 1, 64, p3_h, p3_w,
        p3_cv2_2_w, 64, 1, 1,
        p3_cv2_2_bias,
        1, 1, 0, 0,
        1,
        p3_cv2_out, p3_cv2_out_h, p3_cv2_out_w);
    
    // P3: cv3 Sequential (class prediction)
    // [0]: Conv (64->80, 3x3) + BN + SiLU
    conv_block_nchw_f32(
        p3, 1, p3_c, p3_h, p3_w,
        p3_cv3_0_w, p3_cv3_0_c_out, 3, 3,
        1, 1, 1, 1,
        p3_cv3_0_gamma, p3_cv3_0_beta,
        p3_cv3_0_mean, p3_cv3_0_var,
        eps,
        p3_cv3_step0, p3_h, p3_w);
    
    // [1]: Conv (80->80, 3x3) + BN + SiLU
    conv_block_nchw_f32(
        p3_cv3_step0, 1, 80, p3_h, p3_w,
        p3_cv3_1_w, 80, 3, 3,
        1, 1, 1, 1,
        p3_cv3_1_gamma, p3_cv3_1_beta,
        p3_cv3_1_mean, p3_cv3_1_var,
        eps,
        p3_cv3_step1, p3_h, p3_w);
    
    // [2]: Conv2d (80->80, 1x1) + bias (BN/SiLU 없음)
    conv2d_nchw_f32(
        p3_cv3_step1, 1, 80, p3_h, p3_w,
        p3_cv3_2_w, 80, 1, 1,
        p3_cv3_2_bias,
        1, 1, 0, 0,
        1,
        p3_cv3_out, p3_cv3_out_h, p3_cv3_out_w);
    
    // P4: cv2 Sequential
    conv_block_nchw_f32(
        p4, 1, p4_c, p4_h, p4_w,
        p4_cv2_0_w, p4_cv2_0_c_out, 3, 3,
        1, 1, 1, 1,
        p4_cv2_0_gamma, p4_cv2_0_beta,
        p4_cv2_0_mean, p4_cv2_0_var,
        eps,
        p4_cv2_step0, p4_h, p4_w);
    
    conv_block_nchw_f32(
        p4_cv2_step0, 1, 64, p4_h, p4_w,
        p4_cv2_1_w, 64, 3, 3,
        1, 1, 1, 1,
        p4_cv2_1_gamma, p4_cv2_1_beta,
        p4_cv2_1_mean, p4_cv2_1_var,
        eps,
        p4_cv2_step1, p4_h, p4_w);
    
    conv2d_nchw_f32(
        p4_cv2_step1, 1, 64, p4_h, p4_w,
        p4_cv2_2_w, 64, 1, 1,
        p4_cv2_2_bias,
        1, 1, 0, 0,
        1,
        p4_cv2_out, p4_cv2_out_h, p4_cv2_out_w);
    
    // P4: cv3 Sequential
    conv_block_nchw_f32(
        p4, 1, p4_c, p4_h, p4_w,
        p4_cv3_0_w, p4_cv3_0_c_out, 3, 3,
        1, 1, 1, 1,
        p4_cv3_0_gamma, p4_cv3_0_beta,
        p4_cv3_0_mean, p4_cv3_0_var,
        eps,
        p4_cv3_step0, p4_h, p4_w);
    
    conv_block_nchw_f32(
        p4_cv3_step0, 1, 80, p4_h, p4_w,
        p4_cv3_1_w, 80, 3, 3,
        1, 1, 1, 1,
        p4_cv3_1_gamma, p4_cv3_1_beta,
        p4_cv3_1_mean, p4_cv3_1_var,
        eps,
        p4_cv3_step1, p4_h, p4_w);
    
    conv2d_nchw_f32(
        p4_cv3_step1, 1, 80, p4_h, p4_w,
        p4_cv3_2_w, 80, 1, 1,
        p4_cv3_2_bias,
        1, 1, 0, 0,
        1,
        p4_cv3_out, p4_cv3_out_h, p4_cv3_out_w);
    
    // P5: cv2 Sequential
    conv_block_nchw_f32(
        p5, 1, p5_c, p5_h, p5_w,
        p5_cv2_0_w, p5_cv2_0_c_out, 3, 3,
        1, 1, 1, 1,
        p5_cv2_0_gamma, p5_cv2_0_beta,
        p5_cv2_0_mean, p5_cv2_0_var,
        eps,
        p5_cv2_step0, p5_h, p5_w);
    
    conv_block_nchw_f32(
        p5_cv2_step0, 1, 64, p5_h, p5_w,
        p5_cv2_1_w, 64, 3, 3,
        1, 1, 1, 1,
        p5_cv2_1_gamma, p5_cv2_1_beta,
        p5_cv2_1_mean, p5_cv2_1_var,
        eps,
        p5_cv2_step1, p5_h, p5_w);
    
    conv2d_nchw_f32(
        p5_cv2_step1, 1, 64, p5_h, p5_w,
        p5_cv2_2_w, 64, 1, 1,
        p5_cv2_2_bias,
        1, 1, 0, 0,
        1,
        p5_cv2_out, p5_cv2_out_h, p5_cv2_out_w);
    
    // P5: cv3 Sequential
    conv_block_nchw_f32(
        p5, 1, p5_c, p5_h, p5_w,
        p5_cv3_0_w, p5_cv3_0_c_out, 3, 3,
        1, 1, 1, 1,
        p5_cv3_0_gamma, p5_cv3_0_beta,
        p5_cv3_0_mean, p5_cv3_0_var,
        eps,
        p5_cv3_step0, p5_h, p5_w);
    
    conv_block_nchw_f32(
        p5_cv3_step0, 1, 80, p5_h, p5_w,
        p5_cv3_1_w, 80, 3, 3,
        1, 1, 1, 1,
        p5_cv3_1_gamma, p5_cv3_1_beta,
        p5_cv3_1_mean, p5_cv3_1_var,
        eps,
        p5_cv3_step1, p5_h, p5_w);
    
    conv2d_nchw_f32(
        p5_cv3_step1, 1, 80, p5_h, p5_w,
        p5_cv3_2_w, 80, 1, 1,
        p5_cv3_2_bias,
        1, 1, 0, 0,
        1,
        p5_cv3_out, p5_cv3_out_h, p5_cv3_out_w);
}
