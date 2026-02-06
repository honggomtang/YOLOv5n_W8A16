#include "conv2d_w8a16.h"
#include <stdint.h>

#ifndef CONV2D_TILE_H
#define CONV2D_TILE_H 8
#endif
#ifndef CONV2D_TILE_W
#define CONV2D_TILE_W 8
#endif
#ifndef CONV2D_OC_BLOCK
#define CONV2D_OC_BLOCK 32
#endif

static int32_t conv2d_acc_int32_w8a16[CONV2D_TILE_H][CONV2D_TILE_W][CONV2D_OC_BLOCK];

static inline int16_t clamp_s16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

void conv2d_nchw_w8a16(
    const int16_t* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const int8_t* w, int32_t c_out, int32_t k_h, int32_t k_w,
    const int32_t* bias_or_null,
    uint32_t multiplier,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    int32_t groups,
    int16_t* y, int32_t h_out, int32_t w_out)
{
    if (groups != 1) return;

    const int32_t tile_h = CONV2D_TILE_H;
    const int32_t tile_w = CONV2D_TILE_W;
    const int32_t oc_block = CONV2D_OC_BLOCK;

    const int32_t safe_oh_min = (pad_h + stride_h - 1) / stride_h;
    const int32_t safe_oh_max = (h_in - k_h + pad_h) / stride_h;
    const int32_t safe_ow_min = (pad_w + stride_w - 1) / stride_w;
    const int32_t safe_ow_max = (w_in - k_w + pad_w) / stride_w;

    const int32_t x_h_stride = w_in;
    const int32_t x_c_stride = h_in * w_in;
    const int32_t w_k_stride = k_w;
    const int32_t w_ic_stride = k_h * k_w;
    const int32_t w_oc_stride = c_in * k_h * k_w;

    if (k_h == 1 && k_w == 1) {
        for (int32_t ni = 0; ni < n; ni++) {
            for (int32_t oh0 = 0; oh0 < h_out; oh0 += tile_h) {
                const int32_t oh_end = oh0 + tile_h < h_out ? oh0 + tile_h : h_out;
                const int32_t th = oh_end - oh0;
                for (int32_t ow0 = 0; ow0 < w_out; ow0 += tile_w) {
                    const int32_t ow_end = ow0 + tile_w < w_out ? ow0 + tile_w : w_out;
                    const int32_t tw = ow_end - ow0;
                    for (int32_t oc0 = 0; oc0 < c_out; oc0 += oc_block) {
                        const int32_t n_oc = oc0 + oc_block <= c_out ? oc_block : c_out - oc0;
                        for (int32_t dh = 0; dh < th; dh++) {
                            for (int32_t dw = 0; dw < tw; dw++) {
                                for (int32_t b = 0; b < n_oc; b++) {
                                    int32_t acc = bias_or_null ? bias_or_null[oc0 + b] : 0;
                                    conv2d_acc_int32_w8a16[dh][dw][b] = acc;
                                }
                            }
                        }
                        /* Packed layout [OC/4, IC, 1, 1]: one uint32_t = 4 consecutive OC for same IC. */
                        const uint32_t* w_p = (const uint32_t*)(const void*)w;
                        const int32_t packed_ic_stride = 1;
                        const int32_t packed_oc_stride = c_in * 1 * 1;
                        for (int32_t ic = 0; ic < c_in; ic++) {
                            const int16_t* x_ch = x + (ni * c_in + ic) * x_c_stride;
                            for (int32_t dh = 0; dh < th; dh++) {
                                const int32_t oh = oh0 + dh;
                                const int16_t* x_row = x_ch + oh * x_h_stride + ow0;
                                /* dw를 2씩 증가: uint32_t로 2개(x0,x1) 읽어 4-packed 가중치와 연산. 주소 4바이트 정렬일 때만 패어 로드. */
                                for (int32_t dw = 0; dw < tw; dw += 2) {
                                    int32_t x0, x1;
                                    const int use_pair = (dw + 1 < tw);
                                    if (use_pair) {
                                        uintptr_t addr = (uintptr_t)(const void*)(x_row + dw);
                                        if ((addr & 3u) == 0u) {
                                            uint32_t pair = *(const uint32_t*)(const void*)addr;
                                            x0 = (int32_t)(int16_t)(pair & 0xFFFFu);
                                            x1 = (int32_t)(int16_t)(pair >> 16);
                                        } else {
                                            x0 = (int32_t)x_row[dw];
                                            x1 = (int32_t)x_row[dw + 1];
                                        }
                                    } else {
                                        x0 = (int32_t)x_row[dw];
                                        x1 = 0;
                                    }
                                    for (int32_t b4 = 0; b4 < n_oc; b4 += 4) {
                                        uint32_t p = w_p[(oc0 / 4 + b4 / 4) * packed_oc_stride + ic * packed_ic_stride];
                                        int32_t w0 = (int32_t)(int8_t)(p & 0xFF);
                                        int32_t w1 = (int32_t)(int8_t)((p >> 8) & 0xFF);
                                        int32_t w2 = (int32_t)(int8_t)((p >> 16) & 0xFF);
                                        int32_t w3 = (int32_t)(int8_t)((p >> 24) & 0xFF);
                                        conv2d_acc_int32_w8a16[dh][dw][b4] += x0 * w0;
                                        if (b4 + 1 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 1] += x0 * w1;
                                        if (b4 + 2 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 2] += x0 * w2;
                                        if (b4 + 3 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 3] += x0 * w3;
                                        if (use_pair) {
                                            conv2d_acc_int32_w8a16[dh][dw + 1][b4] += x1 * w0;
                                            if (b4 + 1 < n_oc) conv2d_acc_int32_w8a16[dh][dw + 1][b4 + 1] += x1 * w1;
                                            if (b4 + 2 < n_oc) conv2d_acc_int32_w8a16[dh][dw + 1][b4 + 2] += x1 * w2;
                                            if (b4 + 3 < n_oc) conv2d_acc_int32_w8a16[dh][dw + 1][b4 + 3] += x1 * w3;
                                        }
                                    }
                                }
                            }
                        }
                        for (int32_t dh = 0; dh < th; dh++) {
                            const int32_t oh = oh0 + dh;
                            for (int32_t dw = 0; dw < tw; dw++) {
                                const int32_t ow = ow0 + dw;
                                const int32_t y_off = (ni * c_out + oc0) * h_out * w_out + oh * w_out + ow;
                                for (int32_t b = 0; b < n_oc; b++) {
                                    int32_t acc = conv2d_acc_int32_w8a16[dh][dw][b];
                                    y[y_off + b * h_out * w_out] = clamp_s16((int32_t)(((int64_t)acc * multiplier + 32768) >> 16));
                                }
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    for (int32_t ni = 0; ni < n; ni++) {
        for (int32_t oh0 = 0; oh0 < h_out; oh0 += tile_h) {
            const int32_t oh_end = oh0 + tile_h < h_out ? oh0 + tile_h : h_out;
            const int32_t th = oh_end - oh0;
            for (int32_t ow0 = 0; ow0 < w_out; ow0 += tile_w) {
                const int32_t ow_end = ow0 + tile_w < w_out ? ow0 + tile_w : w_out;
                const int32_t tw = ow_end - ow0;

                for (int32_t oc0 = 0; oc0 < c_out; oc0 += oc_block) {
                    const int32_t n_oc = oc0 + oc_block <= c_out ? oc_block : c_out - oc0;

                    for (int32_t dh = 0; dh < th; dh++) {
                        for (int32_t dw = 0; dw < tw; dw++) {
                            for (int32_t b = 0; b < n_oc; b++) {
                                conv2d_acc_int32_w8a16[dh][dw][b] = bias_or_null ? bias_or_null[oc0 + b] : 0;
                            }
                        }
                    }

                    const int32_t tile_is_safe = (oh0 >= safe_oh_min && oh_end <= safe_oh_max &&
                                                  ow0 >= safe_ow_min && ow_end <= safe_ow_max);
                    const uint32_t* w_p = (const uint32_t*)(const void*)w;
                    const int32_t packed_oc_stride = c_in * k_h * k_w;
                    const int32_t packed_ic_stride = k_h * k_w;

                    for (int32_t ic = 0; ic < c_in; ic++) {
                        for (int32_t b4 = 0; b4 < n_oc; b4 += 4) {
                            const int32_t og = oc0 / 4 + b4 / 4;
                            if (tile_is_safe) {
                                for (int32_t dh = 0; dh < th; dh++) {
                                    const int32_t oh = oh0 + dh;
                                    const int32_t ih0 = oh * stride_h - pad_h;
                                    for (int32_t dw = 0; dw < tw; dw++) {
                                        const int32_t ow = ow0 + dw;
                                        const int32_t iw0 = ow * stride_w - pad_w;
                                        const int16_t* x_base = x + (ni * c_in + ic) * x_c_stride + ih0 * x_h_stride + iw0;
                                        for (int32_t kh = 0; kh < k_h; kh++) {
                                            const int16_t* x_row = x_base + kh * x_h_stride;
                                            /* kw 루프: 가능하면 uint32_t로 2개 읽어 연산 (정렬/홀수 처리) */
                                            int32_t kw = 0;
                                            for (; kw + 1 < k_w; kw += 2) {
                                                uintptr_t addr = (uintptr_t)(const void*)x_row;
                                                int32_t x0, x1;
                                                if ((addr & 3u) == 0u) {
                                                    uint32_t pair = *(const uint32_t*)(const void*)x_row;
                                                    x0 = (int32_t)(int16_t)(pair & 0xFFFFu);
                                                    x1 = (int32_t)(int16_t)(pair >> 16);
                                                } else {
                                                    x0 = (int32_t)x_row[0];
                                                    x1 = (int32_t)x_row[1];
                                                }
                                                x_row += 2;
                                                uint32_t p0 = w_p[og * packed_oc_stride + ic * packed_ic_stride + kh * k_w + kw];
                                                uint32_t p1 = w_p[og * packed_oc_stride + ic * packed_ic_stride + kh * k_w + kw + 1];
                                                int32_t w0 = (int32_t)(int8_t)(p0 & 0xFF);
                                                int32_t w1 = (int32_t)(int8_t)((p0 >> 8) & 0xFF);
                                                int32_t w2 = (int32_t)(int8_t)((p0 >> 16) & 0xFF);
                                                int32_t w3 = (int32_t)(int8_t)((p0 >> 24) & 0xFF);
                                                conv2d_acc_int32_w8a16[dh][dw][b4] += x0 * w0;
                                                if (b4 + 1 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 1] += x0 * w1;
                                                if (b4 + 2 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 2] += x0 * w2;
                                                if (b4 + 3 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 3] += x0 * w3;
                                                w0 = (int32_t)(int8_t)(p1 & 0xFF);
                                                w1 = (int32_t)(int8_t)((p1 >> 8) & 0xFF);
                                                w2 = (int32_t)(int8_t)((p1 >> 16) & 0xFF);
                                                w3 = (int32_t)(int8_t)((p1 >> 24) & 0xFF);
                                                conv2d_acc_int32_w8a16[dh][dw][b4] += x1 * w0;
                                                if (b4 + 1 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 1] += x1 * w1;
                                                if (b4 + 2 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 2] += x1 * w2;
                                                if (b4 + 3 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 3] += x1 * w3;
                                            }
                                            for (; kw < k_w; kw++) {
                                                int32_t x_val = (int32_t)(*x_row++);
                                                uint32_t p = w_p[og * packed_oc_stride + ic * packed_ic_stride + kh * k_w + kw];
                                                int32_t w0 = (int32_t)(int8_t)(p & 0xFF);
                                                int32_t w1 = (int32_t)(int8_t)((p >> 8) & 0xFF);
                                                int32_t w2 = (int32_t)(int8_t)((p >> 16) & 0xFF);
                                                int32_t w3 = (int32_t)(int8_t)((p >> 24) & 0xFF);
                                                conv2d_acc_int32_w8a16[dh][dw][b4] += x_val * w0;
                                                if (b4 + 1 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 1] += x_val * w1;
                                                if (b4 + 2 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 2] += x_val * w2;
                                                if (b4 + 3 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 3] += x_val * w3;
                                            }
                                        }
                                    }
                                }
                            } else {
                                for (int32_t dh = 0; dh < th; dh++) {
                                    const int32_t oh = oh0 + dh;
                                    for (int32_t dw = 0; dw < tw; dw++) {
                                        const int32_t ow = ow0 + dw;
                                        for (int32_t kh = 0; kh < k_h; kh++) {
                                            const int32_t ih = oh * stride_h - pad_h + kh;
                                            if ((uint32_t)ih >= (uint32_t)h_in) continue;
                                            for (int32_t kw = 0; kw < k_w; kw++) {
                                                const int32_t iw = ow * stride_w - pad_w + kw;
                                                if ((uint32_t)iw >= (uint32_t)w_in) continue;
                                                int32_t x_val = (int32_t)x[(ni * c_in + ic) * x_c_stride + ih * x_h_stride + iw];
                                                uint32_t p = w_p[og * packed_oc_stride + ic * packed_ic_stride + kh * k_w + kw];
                                                int32_t w0 = (int32_t)(int8_t)(p & 0xFF);
                                                int32_t w1 = (int32_t)(int8_t)((p >> 8) & 0xFF);
                                                int32_t w2 = (int32_t)(int8_t)((p >> 16) & 0xFF);
                                                int32_t w3 = (int32_t)(int8_t)((p >> 24) & 0xFF);
                                                conv2d_acc_int32_w8a16[dh][dw][b4] += x_val * w0;
                                                if (b4 + 1 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 1] += x_val * w1;
                                                if (b4 + 2 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 2] += x_val * w2;
                                                if (b4 + 3 < n_oc) conv2d_acc_int32_w8a16[dh][dw][b4 + 3] += x_val * w3;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    for (int32_t dh = 0; dh < th; dh++) {
                        const int32_t oh = oh0 + dh;
                        for (int32_t dw = 0; dw < tw; dw++) {
                            const int32_t ow = ow0 + dw;
                            const int32_t y_row_off = (ni * c_out + oc0) * h_out * w_out + oh * w_out + ow;
                            for (int32_t b = 0; b < n_oc; b++) {
                                int32_t acc = conv2d_acc_int32_w8a16[dh][dw][b];
                                y[y_row_off + b * h_out * w_out] = clamp_s16((int32_t)(((int64_t)acc * multiplier + 32768) >> 16));
                            }
                        }
                    }
                }
            }
        }
    }
}

/* FP32 경로 */
static float conv2d_acc_buf_w8a16[CONV2D_TILE_H][CONV2D_TILE_W][CONV2D_OC_BLOCK];

void conv2d_nchw_f32_w8a16(
    const float* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const float* w, int32_t c_out, int32_t k_h, int32_t k_w,
    const float* bias_or_null,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    int32_t groups,
    float* y, int32_t h_out, int32_t w_out)
{
    if (groups != 1) {
        return;
    }

    const int32_t tile_h = CONV2D_TILE_H;
    const int32_t tile_w = CONV2D_TILE_W;
    const int32_t oc_block = CONV2D_OC_BLOCK;

    const int32_t safe_oh_min = (pad_h + stride_h - 1) / stride_h;
    const int32_t safe_oh_max = (h_in - k_h + pad_h) / stride_h;
    const int32_t safe_ow_min = (pad_w + stride_w - 1) / stride_w;
    const int32_t safe_ow_max = (w_in - k_w + pad_w) / stride_w;

    const int32_t x_h_stride = w_in;
    const int32_t x_c_stride = h_in * w_in;
    const int32_t w_k_stride = k_w;
    const int32_t w_ic_stride = k_h * k_w;
    const int32_t w_oc_stride = c_in * k_h * k_w;

    for (int32_t ni = 0; ni < n; ni++) {
        for (int32_t oh0 = 0; oh0 < h_out; oh0 += tile_h) {
            const int32_t oh_end = oh0 + tile_h < h_out ? oh0 + tile_h : h_out;
            const int32_t th = oh_end - oh0;
            for (int32_t ow0 = 0; ow0 < w_out; ow0 += tile_w) {
                const int32_t ow_end = ow0 + tile_w < w_out ? ow0 + tile_w : w_out;
                const int32_t tw = ow_end - ow0;

                for (int32_t oc0 = 0; oc0 < c_out; oc0 += oc_block) {
                    const int32_t n_oc = oc0 + oc_block <= c_out ? oc_block : c_out - oc0;

                    for (int32_t dh = 0; dh < th; dh++) {
                        for (int32_t dw = 0; dw < tw; dw++) {
                            for (int32_t b = 0; b < n_oc; b++) {
                                conv2d_acc_buf_w8a16[dh][dw][b] = bias_or_null ? bias_or_null[oc0 + b] : 0.0f;
                            }
                        }
                    }

                    const int32_t tile_is_safe = (oh0 >= safe_oh_min && oh_end <= safe_oh_max &&
                                                  ow0 >= safe_ow_min && ow_end <= safe_ow_max);

                    for (int32_t ic = 0; ic < c_in; ic++) {
                        for (int32_t b = 0; b < n_oc; b++) {
                            const float* w_base = w + (oc0 + b) * w_oc_stride + ic * w_ic_stride;

                            if (tile_is_safe) {
                                for (int32_t dh = 0; dh < th; dh++) {
                                    const int32_t oh = oh0 + dh;
                                    const int32_t ih0 = oh * stride_h - pad_h;
                                    for (int32_t dw = 0; dw < tw; dw++) {
                                        const int32_t ow = ow0 + dw;
                                        const int32_t iw0 = ow * stride_w - pad_w;
                                        const float* x_base = x + (ni * c_in + ic) * x_c_stride + ih0 * x_h_stride + iw0;
                                        float contrib = 0.0f;
                                        for (int32_t kh = 0; kh < k_h; kh++) {
                                            const float* x_row = x_base + kh * x_h_stride;
                                            const float* w_row = w_base + kh * w_k_stride;
                                            for (int32_t kw = 0; kw < k_w; kw++) {
                                                contrib += (*x_row++) * (*w_row++);
                                            }
                                        }
                                        float* acc_ptr = &conv2d_acc_buf_w8a16[dh][dw][0];
                                        acc_ptr[b] += contrib;
                                    }
                                }
                            } else {
                                for (int32_t dh = 0; dh < th; dh++) {
                                    const int32_t oh = oh0 + dh;
                                    for (int32_t dw = 0; dw < tw; dw++) {
                                        const int32_t ow = ow0 + dw;
                                        const int32_t in_safe = (oh >= safe_oh_min && oh < safe_oh_max &&
                                                                ow >= safe_ow_min && ow < safe_ow_max);
                                        float contrib;
                                        if (in_safe) {
                                            const int32_t ih0 = oh * stride_h - pad_h;
                                            const int32_t iw0 = ow * stride_w - pad_w;
                                            const float* x_base = x + (ni * c_in + ic) * x_c_stride + ih0 * x_h_stride + iw0;
                                            contrib = 0.0f;
                                            for (int32_t kh = 0; kh < k_h; kh++) {
                                                const float* x_row = x_base + kh * x_h_stride;
                                                const float* w_row = w_base + kh * w_k_stride;
                                                for (int32_t kw = 0; kw < k_w; kw++) {
                                                    contrib += (*x_row++) * (*w_row++);
                                                }
                                            }
                                        } else {
                                            const int32_t oc = oc0 + b;
                                            contrib = 0.0f;
                                            for (int32_t kh = 0; kh < k_h; kh++) {
                                                const int32_t ih = oh * stride_h - pad_h + kh;
                                                if ((uint32_t)ih >= (uint32_t)h_in) continue;
                                                for (int32_t kw = 0; kw < k_w; kw++) {
                                                    const int32_t iw = ow * stride_w - pad_w + kw;
                                                    if ((uint32_t)iw >= (uint32_t)w_in) continue;
                                                    const float* x_ptr = x + (ni * c_in + ic) * x_c_stride + ih * x_h_stride + iw;
                                                    const float* w_ptr = w + oc * w_oc_stride + ic * w_ic_stride + kh * w_k_stride + kw;
                                                    contrib += (*x_ptr) * (*w_ptr);
                                                }
                                            }
                                        }
                                        float* acc_ptr = &conv2d_acc_buf_w8a16[dh][dw][0];
                                        acc_ptr[b] += contrib;
                                    }
                                }
                            }
                        }
                    }

                    for (int32_t dh = 0; dh < th; dh++) {
                        const int32_t oh = oh0 + dh;
                        for (int32_t dw = 0; dw < tw; dw++) {
                            const int32_t ow = ow0 + dw;
                            const int32_t y_row_off = (ni * c_out + oc0) * h_out * w_out + oh * w_out + ow;
                            for (int32_t b = 0; b < n_oc; b++) {
                                y[y_row_off + b * h_out * w_out] = conv2d_acc_buf_w8a16[dh][dw][b];
                            }
                        }
                    }
                }
            }
        }
    }
}

/* FP32 input path */
void conv2d_nchw_f32_w8_w8a16(
    const float* x, int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const int8_t* w, float scale, int32_t c_out, int32_t k_h, int32_t k_w,
    const float* bias_or_null,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    int32_t groups,
    float* y, int32_t h_out, int32_t w_out)
{
    if (groups != 1) return;

    const int32_t tile_h = CONV2D_TILE_H;
    const int32_t tile_w = CONV2D_TILE_W;
    const int32_t oc_block = CONV2D_OC_BLOCK;

    const int32_t safe_oh_min = (pad_h + stride_h - 1) / stride_h;
    const int32_t safe_oh_max = (h_in - k_h + pad_h) / stride_h;
    const int32_t safe_ow_min = (pad_w + stride_w - 1) / stride_w;
    const int32_t safe_ow_max = (w_in - k_w + pad_w) / stride_w;

    const int32_t x_h_stride = w_in;
    const int32_t x_c_stride = h_in * w_in;
    const int32_t w_k_stride = k_w;
    const int32_t w_ic_stride = k_h * k_w;
    const int32_t w_oc_stride = c_in * k_h * k_w;

    if (k_h == 1 && k_w == 1) {
        for (int32_t ni = 0; ni < n; ni++) {
            for (int32_t oh0 = 0; oh0 < h_out; oh0 += tile_h) {
                const int32_t oh_end = oh0 + tile_h < h_out ? oh0 + tile_h : h_out;
                const int32_t th = oh_end - oh0;
                for (int32_t ow0 = 0; ow0 < w_out; ow0 += tile_w) {
                    const int32_t ow_end = ow0 + tile_w < w_out ? ow0 + tile_w : w_out;
                    const int32_t tw = ow_end - ow0;
                    for (int32_t oc0 = 0; oc0 < c_out; oc0 += oc_block) {
                        const int32_t n_oc = oc0 + oc_block <= c_out ? oc_block : c_out - oc0;
                        for (int32_t dh = 0; dh < th; dh++) {
                            for (int32_t dw = 0; dw < tw; dw++) {
                                for (int32_t b = 0; b < n_oc; b++)
                                    conv2d_acc_buf_w8a16[dh][dw][b] = bias_or_null ? bias_or_null[oc0 + b] : 0.0f;
                            }
                        }
                        for (int32_t ic = 0; ic < c_in; ic++) {
                            const float* x_ch = x + (ni * c_in + ic) * x_c_stride;
                            for (int32_t dh = 0; dh < th; dh++) {
                                const int32_t oh = oh0 + dh;
                                for (int32_t dw = 0; dw < tw; dw++) {
                                    const int32_t ow = ow0 + dw;
                                    float x_val = x_ch[oh * x_h_stride + ow];
                                    for (int32_t b = 0; b < n_oc; b++)
                                        conv2d_acc_buf_w8a16[dh][dw][b] += x_val * (float)w[(oc0 + b) * w_oc_stride + ic] * scale;
                                }
                            }
                        }
                        for (int32_t dh = 0; dh < th; dh++) {
                            const int32_t oh = oh0 + dh;
                            for (int32_t dw = 0; dw < tw; dw++) {
                                const int32_t ow = ow0 + dw;
                                const int32_t y_off = (ni * c_out + oc0) * h_out * w_out + oh * w_out + ow;
                                for (int32_t b = 0; b < n_oc; b++)
                                    y[y_off + b * h_out * w_out] = conv2d_acc_buf_w8a16[dh][dw][b];
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    for (int32_t ni = 0; ni < n; ni++) {
        for (int32_t oh0 = 0; oh0 < h_out; oh0 += tile_h) {
            const int32_t oh_end = oh0 + tile_h < h_out ? oh0 + tile_h : h_out;
            const int32_t th = oh_end - oh0;
            for (int32_t ow0 = 0; ow0 < w_out; ow0 += tile_w) {
                const int32_t ow_end = ow0 + tile_w < w_out ? ow0 + tile_w : w_out;
                const int32_t tw = ow_end - ow0;

                for (int32_t oc0 = 0; oc0 < c_out; oc0 += oc_block) {
                    const int32_t n_oc = oc0 + oc_block <= c_out ? oc_block : c_out - oc0;

                    for (int32_t dh = 0; dh < th; dh++) {
                        for (int32_t dw = 0; dw < tw; dw++) {
                            for (int32_t b = 0; b < n_oc; b++) {
                                conv2d_acc_buf_w8a16[dh][dw][b] = bias_or_null ? bias_or_null[oc0 + b] : 0.0f;
                            }
                        }
                    }

                    const int32_t tile_is_safe = (oh0 >= safe_oh_min && oh_end <= safe_oh_max &&
                                                  ow0 >= safe_ow_min && ow_end <= safe_ow_max);

                    for (int32_t ic = 0; ic < c_in; ic++) {
                        for (int32_t b = 0; b < n_oc; b++) {
                            const int8_t* w_base = w + (oc0 + b) * w_oc_stride + ic * w_ic_stride;
                            float local_w[36];
                            const int32_t k_size = k_h * k_w;
                            const int8_t* w_src = w_base;
                            int32_t i = 0;
                            if (((uintptr_t)w_src & 3u) == 0) {
                                while (i + 8 <= k_size) {
                                    uint32_t w4a = *(const uint32_t*)w_src; w_src += 4;
                                    local_w[i++] = (float)(int8_t)(w4a & 0xFF) * scale;
                                    local_w[i++] = (float)(int8_t)((w4a >> 8) & 0xFF) * scale;
                                    local_w[i++] = (float)(int8_t)((w4a >> 16) & 0xFF) * scale;
                                    local_w[i++] = (float)(int8_t)((w4a >> 24) & 0xFF) * scale;
                                    uint32_t w4b = *(const uint32_t*)w_src; w_src += 4;
                                    local_w[i++] = (float)(int8_t)(w4b & 0xFF) * scale;
                                    local_w[i++] = (float)(int8_t)((w4b >> 8) & 0xFF) * scale;
                                    local_w[i++] = (float)(int8_t)((w4b >> 16) & 0xFF) * scale;
                                    local_w[i++] = (float)(int8_t)((w4b >> 24) & 0xFF) * scale;
                                }
                                while (i + 4 <= k_size) {
                                    uint32_t w4 = *(const uint32_t*)w_src; w_src += 4;
                                    local_w[i++] = (float)(int8_t)(w4 & 0xFF) * scale;
                                    local_w[i++] = (float)(int8_t)((w4 >> 8) & 0xFF) * scale;
                                    local_w[i++] = (float)(int8_t)((w4 >> 16) & 0xFF) * scale;
                                    local_w[i++] = (float)(int8_t)((w4 >> 24) & 0xFF) * scale;
                                }
                            }
                            for (; i < k_size; i++)
                                local_w[i] = (float)(*w_src++) * scale;

                            if (tile_is_safe) {
                                for (int32_t dh = 0; dh < th; dh++) {
                                    const int32_t oh = oh0 + dh;
                                    const int32_t ih0 = oh * stride_h - pad_h;
                                    for (int32_t dw = 0; dw < tw; dw++) {
                                        const int32_t ow = ow0 + dw;
                                        const int32_t iw0 = ow * stride_w - pad_w;
                                        const float* x_base = x + (ni * c_in + ic) * x_c_stride + ih0 * x_h_stride + iw0;
                                        float contrib = 0.0f;
                                        for (int32_t kh = 0; kh < k_h; kh++) {
                                            const float* x_row = x_base + kh * x_h_stride;
                                            const float* lw_row = local_w + kh * k_w;
                                            for (int32_t kw = 0; kw < k_w; kw++)
                                                contrib += (*x_row++) * lw_row[kw];
                                        }
                                        float* acc_ptr = &conv2d_acc_buf_w8a16[dh][dw][0];
                                        acc_ptr[b] += contrib;
                                    }
                                }
                            } else {
                                for (int32_t dh = 0; dh < th; dh++) {
                                    const int32_t oh = oh0 + dh;
                                    for (int32_t dw = 0; dw < tw; dw++) {
                                        const int32_t ow = ow0 + dw;
                                        const int32_t in_safe = (oh >= safe_oh_min && oh < safe_oh_max &&
                                                                ow >= safe_ow_min && ow < safe_ow_max);
                                        float contrib = 0.0f;
                                        if (in_safe) {
                                            const int32_t ih0 = oh * stride_h - pad_h;
                                            const int32_t iw0 = ow * stride_w - pad_w;
                                            const float* x_base = x + (ni * c_in + ic) * x_c_stride + ih0 * x_h_stride + iw0;
                                            for (int32_t kh = 0; kh < k_h; kh++) {
                                                const float* x_row = x_base + kh * x_h_stride;
                                                const float* lw_row = local_w + kh * k_w;
                                                for (int32_t kw = 0; kw < k_w; kw++)
                                                    contrib += (*x_row++) * lw_row[kw];
                                            }
                                        } else {
                                            for (int32_t kh = 0; kh < k_h; kh++) {
                                                const int32_t ih = oh * stride_h - pad_h + kh;
                                                if ((uint32_t)ih >= (uint32_t)h_in) continue;
                                                for (int32_t kw = 0; kw < k_w; kw++) {
                                                    const int32_t iw = ow * stride_w - pad_w + kw;
                                                    if ((uint32_t)iw >= (uint32_t)w_in) continue;
                                                    const float* x_ptr = x + (ni * c_in + ic) * x_c_stride + ih * x_h_stride + iw;
                                                    contrib += (*x_ptr) * local_w[kh * k_w + kw];
                                                }
                                            }
                                        }
                                        float* acc_ptr = &conv2d_acc_buf_w8a16[dh][dw][0];
                                        acc_ptr[b] += contrib;
                                    }
                                }
                            }
                        }
                    }

                    for (int32_t dh = 0; dh < th; dh++) {
                        const int32_t oh = oh0 + dh;
                        for (int32_t dw = 0; dw < tw; dw++) {
                            const int32_t ow = ow0 + dw;
                            const int32_t y_row_off = (ni * c_out + oc0) * h_out * w_out + oh * w_out + ow;
                            for (int32_t b = 0; b < n_oc; b++) {
                                y[y_row_off + b * h_out * w_out] = conv2d_acc_buf_w8a16[dh][dw][b];
                            }
                        }
                    }
                }
            }
        }
    }
}
