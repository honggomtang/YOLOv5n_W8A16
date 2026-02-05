#include "upsample_w8a16.h"
#include "../utils/timing.h"

void upsample_nearest2x_nchw_w8a16(
    const int16_t* x, int32_t n, int32_t c, int32_t h, int32_t w,
    int16_t* y)
{
    const int32_t out_h = h * 2;
    const int32_t out_w = w * 2;

    for (int32_t ni = 0; ni < n; ni++) {
        for (int32_t ci = 0; ci < c; ci++) {
            for (int32_t ih = 0; ih < h; ih++) {
                for (int32_t iw = 0; iw < w; iw++) {
                    const int32_t in_idx = ((ni * c + ci) * h + ih) * w + iw;
                    const int16_t val = x[in_idx];

                    const int32_t oh0 = ih * 2;
                    const int32_t oh1 = ih * 2 + 1;
                    const int32_t ow0 = iw * 2;
                    const int32_t ow1 = iw * 2 + 1;

                    y[((ni * c + ci) * out_h + oh0) * out_w + ow0] = val;
                    y[((ni * c + ci) * out_h + oh0) * out_w + ow1] = val;
                    y[((ni * c + ci) * out_h + oh1) * out_w + ow0] = val;
                    y[((ni * c + ci) * out_h + oh1) * out_w + ow1] = val;
                }
            }
        }
    }
}

void upsample_nearest2x_nchw_f32_w8a16(
    const float* x, int32_t n, int32_t c, int32_t h, int32_t w,
    float* y)
{
    yolo_timing_begin("upsample");
    const int32_t out_h = h * 2;
    const int32_t out_w = w * 2;

    for (int32_t ni = 0; ni < n; ni++) {
        for (int32_t ci = 0; ci < c; ci++) {
            for (int32_t ih = 0; ih < h; ih++) {
                for (int32_t iw = 0; iw < w; iw++) {
                    const int32_t in_idx = ((ni * c + ci) * h + ih) * w + iw;
                    const float val = x[in_idx];

                    const int32_t oh0 = ih * 2;
                    const int32_t oh1 = ih * 2 + 1;
                    const int32_t ow0 = iw * 2;
                    const int32_t ow1 = iw * 2 + 1;

                    y[((ni * c + ci) * out_h + oh0) * out_w + ow0] = val;
                    y[((ni * c + ci) * out_h + oh0) * out_w + ow1] = val;
                    y[((ni * c + ci) * out_h + oh1) * out_w + ow0] = val;
                    y[((ni * c + ci) * out_h + oh1) * out_w + ow1] = val;
                }
            }
        }
    }
    yolo_timing_end();
}
