#include "conv2d_w8a32.h"
#include <stdint.h>

/* conv2d 최적화 포인트:
 * - 출력 타일링 (기본 8x8)
 * - safe 영역/경계 분리 (경계만 bounds 체크)
 * - 정적 acc 버퍼(BSS) 사용 (bare-metal 스택 절약) */
#ifndef CONV2D_TILE_H
#define CONV2D_TILE_H 8
#endif
#ifndef CONV2D_TILE_W
#define CONV2D_TILE_W 8
#endif
#ifndef CONV2D_OC_BLOCK
#define CONV2D_OC_BLOCK 32
#endif

static float conv2d_acc_buf_w8a32[CONV2D_TILE_H][CONV2D_TILE_W][CONV2D_OC_BLOCK];

void conv2d_nchw_f32_w8a32(
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
                                conv2d_acc_buf_w8a32[dh][dw][b] = bias_or_null ? bias_or_null[oc0 + b] : 0.0f;
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
                                        float* acc_ptr = &conv2d_acc_buf_w8a32[dh][dw][0];
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
                                        float* acc_ptr = &conv2d_acc_buf_w8a32[dh][dw][0];
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
                                y[y_row_off + b * h_out * w_out] = conv2d_acc_buf_w8a32[dh][dw][b];
                            }
                        }
                    }
                }
            }
        }
    }
}

/* W8A32: INT8 weights (per-tensor scale), FP32 compute */
void conv2d_nchw_f32_w8_w8a32(
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
                                    conv2d_acc_buf_w8a32[dh][dw][b] = bias_or_null ? bias_or_null[oc0 + b] : 0.0f;
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
                                        conv2d_acc_buf_w8a32[dh][dw][b] += x_val * (float)w[(oc0 + b) * w_oc_stride + ic] * scale;
                                }
                            }
                        }
                        for (int32_t dh = 0; dh < th; dh++) {
                            const int32_t oh = oh0 + dh;
                            for (int32_t dw = 0; dw < tw; dw++) {
                                const int32_t ow = ow0 + dw;
                                const int32_t y_off = (ni * c_out + oc0) * h_out * w_out + oh * w_out + ow;
                                for (int32_t b = 0; b < n_oc; b++)
                                    y[y_off + b * h_out * w_out] = conv2d_acc_buf_w8a32[dh][dw][b];
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
                                conv2d_acc_buf_w8a32[dh][dw][b] = bias_or_null ? bias_or_null[oc0 + b] : 0.0f;
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
                                        float* acc_ptr = &conv2d_acc_buf_w8a32[dh][dw][0];
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
                                        float* acc_ptr = &conv2d_acc_buf_w8a32[dh][dw][0];
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
                                y[y_row_off + b * h_out * w_out] = conv2d_acc_buf_w8a32[dh][dw][b];
                            }
                        }
                    }
                }
            }
        }
    }
}
