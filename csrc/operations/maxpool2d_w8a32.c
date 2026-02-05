#include "maxpool2d_w8a32.h"

void maxpool2d_nchw_f32_w8a32(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    int32_t k, int32_t stride, int32_t pad,
    float* y, int32_t out_h, int32_t out_w)
{
    for (int32_t ni = 0; ni < n; ni++) {
        for (int32_t ci = 0; ci < c; ci++) {
            for (int32_t oh = 0; oh < out_h; oh++) {
                for (int32_t ow = 0; ow < out_w; ow++) {
                    float m = -3.402823466e+38f;

                    for (int32_t kh = 0; kh < k; kh++) {
                        for (int32_t kw = 0; kw < k; kw++) {
                            const int32_t ih = oh * stride - pad + kh;
                            const int32_t iw = ow * stride - pad + kw;
                            if ((uint32_t)ih >= (uint32_t)h || (uint32_t)iw >= (uint32_t)w) {
                                continue;
                            }
                            const int32_t idx = ((ni * c + ci) * h + ih) * w + iw;
                            const float v = x[idx];
                            if (v > m) m = v;
                        }
                    }

                    const int32_t oidx = ((ni * c + ci) * out_h + oh) * out_w + ow;
                    y[oidx] = m;
                }
            }
        }
    }
}
