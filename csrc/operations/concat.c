#include "concat.h"

void concat_nchw_f32(
    const float* x1, int32_t c1,
    const float* x2, int32_t c2,
    int32_t n, int32_t h, int32_t w,
    float* y)
{
    const int32_t hw = h * w;
    
    for (int32_t ni = 0; ni < n; ni++) {
        // x1 채널들 복사
        for (int32_t ci = 0; ci < c1; ci++) {
            const int32_t src_base = ((ni * c1 + ci) * h) * w;
            const int32_t dst_base = ((ni * (c1 + c2) + ci) * h) * w;
            for (int32_t i = 0; i < hw; i++) {
                y[dst_base + i] = x1[src_base + i];
            }
        }
        // x2 채널들 복사
        for (int32_t ci = 0; ci < c2; ci++) {
            const int32_t src_base = ((ni * c2 + ci) * h) * w;
            const int32_t dst_base = ((ni * (c1 + c2) + (c1 + ci)) * h) * w;
            for (int32_t i = 0; i < hw; i++) {
                y[dst_base + i] = x2[src_base + i];
            }
        }
    }
}

void concat4_nchw_f32(
    const float* x0, int32_t c0,
    const float* x1, int32_t c1,
    const float* x2, int32_t c2,
    const float* x3, int32_t c3,
    int32_t n, int32_t h, int32_t w,
    float* y)
{
    const int32_t hw = h * w;
    const int32_t c01 = c0 + c1;
    const int32_t c012 = c01 + c2;
    const int32_t c0123 = c012 + c3;

    for (int32_t ni = 0; ni < n; ni++) {
        // x0
        for (int32_t ci = 0; ci < c0; ci++) {
            const int32_t src = ((ni * c0 + ci) * h) * w;
            const int32_t dst = ((ni * c0123 + ci) * h) * w;
            for (int32_t i = 0; i < hw; i++) y[dst + i] = x0[src + i];
        }
        // x1
        for (int32_t ci = 0; ci < c1; ci++) {
            const int32_t src = ((ni * c1 + ci) * h) * w;
            const int32_t dst = ((ni * c0123 + (c0 + ci)) * h) * w;
            for (int32_t i = 0; i < hw; i++) y[dst + i] = x1[src + i];
        }
        // x2
        for (int32_t ci = 0; ci < c2; ci++) {
            const int32_t src = ((ni * c2 + ci) * h) * w;
            const int32_t dst = ((ni * c0123 + (c01 + ci)) * h) * w;
            for (int32_t i = 0; i < hw; i++) y[dst + i] = x2[src + i];
        }
        // x3
        for (int32_t ci = 0; ci < c3; ci++) {
            const int32_t src = ((ni * c3 + ci) * h) * w;
            const int32_t dst = ((ni * c0123 + (c012 + ci)) * h) * w;
            for (int32_t i = 0; i < hw; i++) y[dst + i] = x3[src + i];
        }
    }
}
