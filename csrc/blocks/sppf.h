#ifndef SPPF_H
#define SPPF_H

#include <stdint.h>

// SPPF: cv1 -> maxpool x3 -> concat4 -> cv2
void sppf_nchw_f32(
    const float* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    // cv1 (1x1)
    const float* cv1_w, int32_t cv1_c_out,
    const float* cv1_gamma, const float* cv1_beta,
    const float* cv1_mean, const float* cv1_var,
    // cv2 (1x1)
    const float* cv2_w, int32_t cv2_c_out,
    const float* cv2_gamma, const float* cv2_beta,
    const float* cv2_mean, const float* cv2_var,
    // pool
    int32_t pool_k,
    float eps,
    float* y);

#endif // SPPF_H
