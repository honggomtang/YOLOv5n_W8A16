#ifndef TYPES_W8A16_H
#define TYPES_W8A16_H

#include <stdint.h>

/*
 * W8A16 고정소수점 타입 (Q6.10)
 * - 부호 1비트, 정수 5비트, 소수 10비트
 * - 표현 범위: 약 [-32, +32)
 * - 입력값은 ×2^10 하여 int16_t로 저장
 */
#define Q6_10_FRAC_BITS 10
#define Q6_10_SCALE     (1 << Q6_10_FRAC_BITS)  /* 1024 */

typedef int16_t  activation_t_w8a16;  /* Input/Activation: Q6.10 */
typedef int8_t   weight_t_w8a16;      /* Weight: 기존 int8_t 유지 */
typedef int32_t  accum_t_w8a16;      /* 곱셈 누산기 (16×8→24비트, int32_t) */

/* float → Q6.10 (×1024 후 clamp to int16_t) */
static inline int16_t float_to_q610(float x) {
    float v = x * (float)Q6_10_SCALE;
    if (v > 32767.0f) return 32767;
    if (v < -32768.0f) return -32768;
    return (int16_t)(int32_t)v;
}

/* Q6.10 → float */
static inline float q610_to_float(int16_t x) {
    return (float)x / (float)Q6_10_SCALE;
}

#endif // TYPES_W8A16_H
