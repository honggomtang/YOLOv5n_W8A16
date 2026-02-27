#ifndef TYPES_W8A16_H
#define TYPES_W8A16_H

#include <stdint.h>

#define Q6_10_FRAC_BITS 10
#define Q6_10_SCALE     (1 << Q6_10_FRAC_BITS)

typedef int16_t  activation_t_w8a16;
typedef int8_t   weight_t_w8a16;
typedef int32_t  accum_t_w8a16;

static inline int16_t float_to_q610(float x) {
    float v = x * (float)Q6_10_SCALE;
    if (v > 32767.0f) return 32767;
    if (v < -32768.0f) return -32768;
    return (int16_t)(int32_t)v;
}

static inline float q610_to_float(int16_t x) {
    return (float)x / (float)Q6_10_SCALE;
}

#endif // TYPES_W8A16_H
