#include "conv_w8a16.h"
#include "../operations/conv2d_w8a16.h"
#include "../operations/silu_w8a16.h"
#include "../utils/timing.h"
#if defined(USE_CONV_ACC)
#include "../drivers/conv_acc_driver.h"
#include "../utils/feature_pool.h"
#endif

void conv_block_nchw_w8a16(
    const int16_t* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const int8_t* w, int32_t c_out, int32_t k_h, int32_t k_w,
    const int32_t* bias_or_null, uint32_t multiplier,
    int32_t stride_h, int32_t stride_w, int32_t pad_h, int32_t pad_w,
    int16_t* y, int32_t h_out, int32_t w_out)
{
    yolo_timing_begin("conv2d");
    if (stride_h > 2 || stride_w > 2) {
        conv2d_nchw_w8a16(x, n, c_in, h_in, w_in, w, c_out, k_h, k_w,
                          bias_or_null, multiplier, stride_h, stride_w, pad_h, pad_w, 1,
                          y, h_out, w_out);
#if defined(USE_CONV_ACC) && defined(BARE_METAL)
        yolo_timing_end_with_op("conv2d");
#else
        yolo_timing_end();
#endif
    } else {
#if defined(USE_CONV_ACC) && defined(BARE_METAL)
        int32_t padded_h = h_in + 2 * pad_h;
        int32_t padded_w = w_in + 2 * pad_w;
        uint32_t need = conv_acc_scratch_size(c_in, k_h, k_w, padded_h, padded_w, h_out, w_out);
        void* scratch = feature_pool_scratch_alloc((size_t)need);
        int acc_used = 0;
        if (scratch && need > 0) {
            acc_used = conv_layer_run(x, n, c_in, h_in, w_in, w, c_out, k_h, k_w,
                bias_or_null, multiplier, stride_h, stride_w, pad_h, pad_w,
                y, h_out, w_out, scratch, need);
        } else {
            conv2d_nchw_w8a16(x, n, c_in, h_in, w_in, w, c_out, k_h, k_w,
                bias_or_null, multiplier, stride_h, stride_w, pad_h, pad_w, 1, y, h_out, w_out);
        }
        yolo_timing_end_with_op(acc_used ? "conv2d_acc" : "conv2d");
#else
        conv2d_nchw_w8a16(x, n, c_in, h_in, w_in, w, c_out, k_h, k_w,
                          bias_or_null, multiplier, stride_h, stride_w, pad_h, pad_w, 1,
                          y, h_out, w_out);
        yolo_timing_end();
#endif
    }
    yolo_timing_begin("silu");
    silu_nchw_w8a16(y, n, c_out, h_out, w_out, y);
    yolo_timing_end();
}

void conv_block_nchw_f32_w8a16(
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
        conv2d_nchw_f32_w8_w8a16(x, n, c_in, h_in, w_in,
                           (const int8_t*)w, w_scale, c_out, k_h, k_w,
                           bias, stride_h, stride_w, pad_h, pad_w, 1,
                           y, h_out, w_out);
    } else if (w) {
        conv2d_nchw_f32_w8a16(x, n, c_in, h_in, w_in,
                        (const float*)w, c_out, k_h, k_w,
                        bias, stride_h, stride_w, pad_h, pad_w, 1,
                        y, h_out, w_out);
    }
    yolo_timing_end();
    yolo_timing_begin("silu");
    silu_nchw_f32_w8a16(y, n, c_out, h_out, w_out, y);
    yolo_timing_end();
}
