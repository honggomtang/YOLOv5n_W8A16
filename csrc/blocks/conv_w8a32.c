#include "conv_w8a32.h"
#include "../operations/conv2d_w8a32.h"
#include "../operations/silu_w8a32.h"
#include "../utils/timing.h"

void conv_block_nchw_f32_w8a32(
    const float* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const void* w, float w_scale, int w_is_int8,
    int32_t c_out, int32_t k_h, int32_t k_w,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    const float* bias,
    float* y, int32_t h_out, int32_t w_out)
{
    yolo_timing_begin("conv2d");
    if (w_is_int8 && w) {
        conv2d_nchw_f32_w8_w8a32(x, n, c_in, h_in, w_in,
                           (const int8_t*)w, w_scale, c_out, k_h, k_w,
                           bias, stride_h, stride_w, pad_h, pad_w, 1,
                           y, h_out, w_out);
    } else if (w) {
        conv2d_nchw_f32_w8a32(x, n, c_in, h_in, w_in,
                        (const float*)w, c_out, k_h, k_w,
                        bias, stride_h, stride_w, pad_h, pad_w, 1,
                        y, h_out, w_out);
    }
    yolo_timing_end();
    yolo_timing_begin("silu");
    silu_nchw_f32_w8a32(y, n, c_out, h_out, w_out, y);
    yolo_timing_end();
}
