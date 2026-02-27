#include "silu_w8a16.h"
#include "silu_lut_data.h"
#include <stdint.h>
#include <math.h>


void silu_nchw_w8a16(
    const int16_t* x, int32_t n, int32_t c, int32_t h, int32_t w,
    int16_t* y)
{
    int32_t total = n * c * h * w;
    for (int32_t i = 0; i < total; i++) {
        uint16_t idx = (uint16_t)x[i];
        y[i] = silu_lut_q610[idx];
    }
}

static inline float silu_f32_w8a16(float x) {
    if (!isfinite(x)) {
        return (x > 0.0f) ? 100.0f : 0.0f;
    }
    float s = 1.0f / (1.0f + expf(-x));
    return x * s;
}

void silu_nchw_f32_w8a16(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    float* y)
{
    int32_t total = n * c * h * w;
    for (int32_t i = 0; i < total; i++) {
        y[i] = silu_f32_w8a16(x[i]);
    }
}
