#ifndef CONCAT_W8A16_H
#define CONCAT_W8A16_H

#include <stdint.h>

void concat_nchw_f32_w8a16(
    const float* x1, int32_t c1,
    const float* x2, int32_t c2,
    int32_t n, int32_t h, int32_t w,
    float* y);

void concat_nchw_w8a16(
    const int16_t* x1, int32_t c1,
    const int16_t* x2, int32_t c2,
    int32_t n, int32_t h, int32_t w,
    int16_t* y);

void concat4_nchw_f32_w8a16(
    const float* x0, int32_t c0,
    const float* x1, int32_t c1,
    const float* x2, int32_t c2,
    const float* x3, int32_t c3,
    int32_t n, int32_t h, int32_t w,
    float* y);

void concat4_nchw_w8a16(
    const int16_t* x0, int32_t c0,
    const int16_t* x1, int32_t c1,
    const int16_t* x2, int32_t c2,
    const int16_t* x3, int32_t c3,
    int32_t n, int32_t h, int32_t w,
    int16_t* y);

#endif // CONCAT_W8A16_H
