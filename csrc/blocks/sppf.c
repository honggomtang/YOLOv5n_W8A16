#include "sppf.h"
#include "../operations/conv2d.h"
#include "../operations/bn_silu.h"
#include "../operations/maxpool2d.h"
#include "../operations/concat.h"

void sppf_nchw_f32(
    const float* x, int32_t n, int32_t c_in, int32_t h, int32_t w,
    const float* cv1_w, int32_t cv1_c_out,
    const float* cv1_gamma, const float* cv1_beta,
    const float* cv1_mean, const float* cv1_var,
    const float* cv2_w, int32_t cv2_c_out,
    const float* cv2_gamma, const float* cv2_beta,
    const float* cv2_mean, const float* cv2_var,
    int32_t pool_k,
    float eps,
    float* y)
{
    // 일단 성능 말고 정답부터
    const int32_t pad = pool_k / 2;

    static float x1[1024 * 1024];       // (n, c_, h, w)
    static float y1[1024 * 1024];       // (n, c_, h, w)
    static float y2[1024 * 1024];       // (n, c_, h, w)
    static float y3[1024 * 1024];       // (n, c_, h, w)
    static float cat[1024 * 1024];      // (n, 4*c_, h, w)
    static float cv2_out[1024 * 1024];  // (n, c2, h, w)

    // cv1: 1x1 conv + bn+silu
    conv2d_nchw_f32(x, n, c_in, h, w,
                    cv1_w, cv1_c_out, 1, 1,
                    0,
                    1, 1,
                    0, 0,
                    1,
                    x1, h, w);
    bn_silu_nchw_f32(x1, n, cv1_c_out, h, w,
                     cv1_gamma, cv1_beta, cv1_mean, cv1_var,
                     eps,
                     x1);

    // y1 = m(x1), y2 = m(y1), y3 = m(y2)
    maxpool2d_nchw_f32(x1, n, cv1_c_out, h, w, pool_k, 1, pad, y1, h, w);
    maxpool2d_nchw_f32(y1, n, cv1_c_out, h, w, pool_k, 1, pad, y2, h, w);
    maxpool2d_nchw_f32(y2, n, cv1_c_out, h, w, pool_k, 1, pad, y3, h, w);

    // cat = [x1, y1, y2, y3]
    concat4_nchw_f32(x1, cv1_c_out, y1, cv1_c_out, y2, cv1_c_out, y3, cv1_c_out,
                     n, h, w,
                     cat);

    // cv2: 1x1 conv + bn+silu
    conv2d_nchw_f32(cat, n, 4 * cv1_c_out, h, w,
                    cv2_w, cv2_c_out, 1, 1,
                    0,
                    1, 1,
                    0, 0,
                    1,
                    cv2_out, h, w);
    bn_silu_nchw_f32(cv2_out, n, cv2_c_out, h, w,
                     cv2_gamma, cv2_beta, cv2_mean, cv2_var,
                     eps,
                     y);
}
