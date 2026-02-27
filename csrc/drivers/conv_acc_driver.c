#include "conv_acc_driver.h"
#include <string.h>
#include <math.h>

#if defined(BARE_METAL)
#include "xparameters.h"
#include "xaxidma.h"
#include "xil_io.h"
#include "xil_cache.h"
#include "xil_printf.h"
#ifndef XPAR_XAXIDMA_0_DEVICE_ID
#define XPAR_XAXIDMA_0_DEVICE_ID  0
#endif
#endif

#define NUM_CLUSTERS  8
#define NUM_PE        32

void conv_acc_weight_repack(
    const int8_t* w_oc_ic_kh_kw,
    int32_t oc_total, int32_t ic, int32_t kh, int32_t kw,
    int32_t oc_block_index,
    uint32_t* out)
{
    const int32_t oc0 = oc_block_index * NUM_PE;
    const size_t ic_kh_kw = (size_t)ic * (size_t)kh * (size_t)kw;
    const size_t kh_kw = (size_t)kh * (size_t)kw;
    for (int32_t ic_ = 0; ic_ < ic; ic_++) {
        for (int32_t kh_ = 0; kh_ < kh; kh_++) {
            for (int32_t kw_ = 0; kw_ < kw; kw_++) {
                size_t base = (size_t)ic_ * kh_kw + (size_t)kh_ * (size_t)kw + (size_t)kw_;
                for (int32_t c = 0; c < NUM_CLUSTERS; c++) {
                    int32_t g = (oc0 + c * 4) / 4;
                    size_t word_idx = (size_t)g * ic_kh_kw + base;
                    uint32_t word = *(const uint32_t*)(w_oc_ic_kh_kw + word_idx * 4u);
                    for (int32_t pe = 0; pe < 4; pe++) {
                        if (oc0 + c * 4 + pe >= oc_total)
                            word &= ~(0xFFu << (unsigned)(pe * 8));
                    }
                    *out++ = word;
                }
            }
        }
    }
}

void conv_acc_bias_quant(
    const float* bias_float,
    float scale_weight,
    int32_t c_out,
    int32_t* out)
{
    if (!out) return;
    if (scale_weight <= 0.f) {
        for (int i = 0; i < NUM_PE; i++) out[i] = 0;
        return;
    }
    float factor = 1024.0f / scale_weight;
    int32_t k;
    for (k = 0; k < c_out && k < NUM_PE; k++)
        out[k] = (int32_t)roundf(bias_float[k] * factor);
    for (; k < NUM_PE; k++)
        out[k] = 0;
}

void conv_acc_pack_activation_line(
    const int16_t* src,
    int32_t img_width,
    int32_t target_ic,
    int32_t line_idx,
    uint32_t* dst)
{
    (void)line_idx;
    int32_t line_len = img_width * (target_ic / 2);
    for (int32_t i = 0; i < line_len; i++) {
        int32_t c0 = (i * 2) % target_ic;
        int32_t c1 = c0 + 1;
        int32_t col = (i * 2) / target_ic;
        if (c1 >= target_ic) { *dst++ = (uint32_t)(uint16_t)src[col * target_ic + c0]; continue; }
        int16_t a = src[col * target_ic + c0];
        int16_t b = src[col * target_ic + c1];
        *dst++ = (uint32_t)(uint16_t)a | ((uint32_t)(uint16_t)b << 16);
    }
}

static void pack_one_line_nchw(
    const int16_t* x,
    int32_t c_in,
    int32_t padded_w,
    int32_t line_idx,
    int32_t pad_h,
    int32_t pad_w,
    int32_t h_in,
    int32_t w_in,
    uint32_t* dst)
{
    int32_t row = line_idx - pad_h;
    const int32_t x_c_stride = h_in * w_in;
    const int32_t x_h_stride = w_in;
    for (int32_t col = 0; col < padded_w; col++) {
        int32_t ic = col - pad_w;
        for (int32_t ch = 0; ch < c_in; ch += 2) {
            int16_t a = 0, b = 0;
            if (row >= 0 && row < h_in && ic >= 0 && ic < w_in) {
                a = x[ch * x_c_stride + row * x_h_stride + ic];
                if (ch + 1 < c_in)
                    b = x[(ch + 1) * x_c_stride + row * x_h_stride + ic];
            }
            *dst++ = (uint32_t)(uint16_t)a | ((uint32_t)(uint16_t)b << 16);
        }
    }
}

static void pack_one_line_padded_to_maxw(
    const int16_t* x,
    int32_t c_in,
    int32_t padded_w,
    int32_t line_idx,
    int32_t pad_h,
    int32_t pad_w,
    int32_t h_in,
    int32_t w_in,
    uint32_t* dst)
{
    int32_t line_len = padded_w * (c_in / 2);
    pack_one_line_nchw(x, c_in, padded_w, line_idx, pad_h, pad_w, h_in, w_in, dst);
    for (int32_t i = line_len; i < (int32_t)CONV_ACC_MAX_W_LINE; i++)
        dst[i] = 0;
}

uint32_t conv_acc_scratch_size(int32_t c_in, int32_t k_h, int32_t k_w,
    int32_t padded_h, int32_t padded_w, int32_t h_out, int32_t w_out)
{
    (void)padded_h;
    (void)padded_w;
    (void)h_out;
    uint32_t bias_sz = 128u;
    uint32_t weight_sz = (uint32_t)(c_in * k_h * k_w * NUM_CLUSTERS * 4);
    uint32_t act_sz = (uint32_t)((int32_t)k_h * (int32_t)CONV_ACC_MAX_W_LINE * 4);
    uint32_t out_sz = (uint32_t)(w_out * 16u * 4u);
    return bias_sz + weight_sz + act_sz + out_sz;
}

#if defined(BARE_METAL)
static XAxiDma s_axi_dma;
static int s_dma_ready = 0;

static void ch1_rmw(uint32_t clear_mask, uint32_t val, unsigned int shift) {
    uint32_t r = Xil_In32(CONV_ACC_CH1_ADDR);
    r = (r & ~clear_mask) | ((val << shift) & clear_mask);
    Xil_Out32(CONV_ACC_CH1_ADDR, r);
}

static void conv_acc_set_target_ic(uint32_t val) {
    Xil_Out32(CONV_ACC_GPIO_BASE_0, val);
}
static void conv_acc_set_kernel_size(uint32_t val) {
    ch1_rmw(CONV_ACC_CH1_KERNEL_SIZE_MASK, val & 0xFu, CONV_ACC_CH1_KERNEL_SIZE_SHIFT);
}
static void conv_acc_set_img_width(uint32_t val) {
    ch1_rmw(CONV_ACC_CH1_IMG_WIDTH_MASK, val & 0xFFFFu, CONV_ACC_CH1_IMG_WIDTH_SHIFT);
}
static void conv_acc_set_stride(uint32_t val) {
    ch1_rmw(CONV_ACC_CH1_STRIDE_MASK, val & 0xFu, CONV_ACC_CH1_STRIDE_SHIFT);
}
static void conv_acc_set_act_start(uint32_t val) {
    ch1_rmw(CONV_ACC_CH1_ACT_START_MASK, val & 0x7Fu, CONV_ACC_CH1_ACT_START_SHIFT);
}
static void conv_acc_set_start_load(uint32_t val) {
    ch1_rmw(CONV_ACC_CH1_START_LOAD_MASK, val ? 1u : 0u, CONV_ACC_CH1_START_LOAD_SHIFT);
}
static void conv_acc_set_multiplier(uint32_t val) {
    Xil_Out32(CONV_ACC_CH2_ADDR, val);
}

static int poll_tx_done(void) {
    while (XAxiDma_Busy(&s_axi_dma, XAXIDMA_DMA_TO_DEVICE))
        ;
    return 0;
}

#ifndef XAXIDMA_RX_OFFSET
#define XAXIDMA_RX_OFFSET  0x30u
#endif
#ifndef XAXIDMA_SR_OFFSET
#define XAXIDMA_SR_OFFSET  0x04u
#endif
#ifndef XAXIDMA_ERR_ALL_MASK
#define XAXIDMA_ERR_ALL_MASK  0x770u
#endif
static void dump_dma_rx_status(void) {
    uint32_t addr = (uint32_t)s_axi_dma.RegBase + XAXIDMA_RX_OFFSET + XAXIDMA_SR_OFFSET;
    uint32_t sr = Xil_In32(addr);
    xil_printf("DMA RX SR: 0x%08X (Err bits 0x%03X)\r\n", (unsigned)sr, (unsigned)(sr & XAXIDMA_ERR_ALL_MASK));
}

#define POLL_RX_TIMEOUT  50000000u
static int poll_rx_done(void) {
    uint32_t i;
    for (i = 0; i < POLL_RX_TIMEOUT; i++) {
        if (!XAxiDma_Busy(&s_axi_dma, XAXIDMA_DEVICE_TO_DMA))
            return 0;
    }
    xil_printf("poll_rx_done TIMEOUT after %u iters\r\n", (unsigned)i);
    dump_dma_rx_status();
    return -1;
}
#endif

int conv_acc_run_once(
    uint32_t kernel_size,
    uint32_t target_ic,
    uint16_t img_width,
    uint32_t stride_w,
    uint32_t act_start,
    uint32_t multiplier,
    const uint32_t* bias_buf,
    const uint32_t* weight_buf,
    uint32_t weight_num_words,
    const uint32_t* act_buf,
    uint32_t act_total_words,
    uint32_t w_out,
    uint32_t* out_buf,
    int first_row)
{
#if !defined(BARE_METAL)
    (void)kernel_size; (void)target_ic; (void)img_width; (void)stride_w; (void)act_start; (void)multiplier;
    (void)bias_buf; (void)weight_buf; (void)weight_num_words;
    (void)act_buf; (void)act_total_words; (void)w_out; (void)out_buf; (void)first_row;
    return -1;
#else
    if (!s_dma_ready) return -2;

    conv_acc_set_target_ic(target_ic);
    conv_acc_set_kernel_size(kernel_size);
    conv_acc_set_img_width(img_width);
    conv_acc_set_stride(stride_w);
    conv_acc_set_act_start(act_start > 127u ? 127u : act_start);
    conv_acc_set_multiplier(multiplier);

    if (first_row) {
        conv_acc_set_start_load(1u);
        Xil_DCacheFlushRange((UINTPTR)bias_buf, 128);
        if (XAxiDma_SimpleTransfer(&s_axi_dma, (UINTPTR)bias_buf, 128, XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS)
            return -3;
        poll_tx_done();

        Xil_DCacheFlushRange((UINTPTR)weight_buf, weight_num_words * 4);
        if (XAxiDma_SimpleTransfer(&s_axi_dma, (UINTPTR)weight_buf, weight_num_words * 4, XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS)
            return -4;
        poll_tx_done();
        conv_acc_set_start_load(0u);
    } else {
        conv_acc_set_start_load(0u);
    }

    Xil_DCacheFlushRange((UINTPTR)act_buf, act_total_words * 4);
    if (XAxiDma_SimpleTransfer(&s_axi_dma, (UINTPTR)act_buf, act_total_words * 4, XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS)
        return -5;
    poll_tx_done();

    {
        uint32_t out_num_words = w_out * 16u;
        uint32_t expected_bytes = out_num_words * 4u;
        if (XAxiDma_SimpleTransfer(&s_axi_dma, (UINTPTR)out_buf, expected_bytes, XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS)
            return -6;
        if (poll_rx_done() != 0)
            return -8;
        Xil_DCacheInvalidateRange((UINTPTR)out_buf, out_num_words * 4);
    }
    return 0;
#endif
}

#if defined(BARE_METAL)
int conv_acc_dma_init(void) {
    XAxiDma_Config* cfg = XAxiDma_LookupConfig(XPAR_XAXIDMA_0_DEVICE_ID);
    if (!cfg) return -1;
    if (XAxiDma_CfgInitialize(&s_axi_dma, cfg) != XST_SUCCESS) return -2;
    s_dma_ready = 1;
    return 0;
}
#endif

static void unpack_out_one_row(
    const uint32_t* out_buf,
    int32_t w_out,
    int32_t out_row,
    int32_t oc0, int32_t c_out,
    int16_t* y,
    int32_t h_out)
{
    for (int32_t ow = 0; ow < w_out; ow++) {
        const uint32_t* pix = out_buf + ow * 16;
        for (int32_t ch = 0; ch < 32 && (oc0 + ch) < c_out; ch++) {
            uint32_t w = pix[ch / 2];
            int16_t v = (ch & 1) ? (int16_t)(w >> 16) : (int16_t)(w & 0xFFFFu);
            y[(oc0 + ch) * h_out * w_out + out_row * w_out + ow] = v;
        }
    }
}

int conv_acc_layer_run(
    const int16_t* x,
    int32_t n, int32_t c_in, int32_t h_in, int32_t w_in,
    const int8_t* w,
    const int32_t* bias,
    uint32_t multiplier,
    int32_t c_out, int32_t k_h, int32_t k_w,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    int16_t* y,
    int32_t h_out, int32_t w_out,
    void* scratch_buf,
    uint32_t scratch_size)
{
    if ((c_in & 1) != 0) return -1;
    if (stride_h > 2 || stride_w > 2) return -2;
    int32_t padded_h = h_in + 2 * pad_h;
    int32_t padded_w = w_in + 2 * pad_w;
    if (padded_w < (int32_t)k_w) return -3;
    if ((uint32_t)(padded_w * (c_in / 2)) > CONV_ACC_MAX_W_LINE) return -7;
    if (!scratch_buf) return -4;

    uint32_t need = conv_acc_scratch_size(c_in, k_h, k_w, padded_h, padded_w, h_out, w_out);
    if (scratch_size < need) return -5;

#if !defined(BARE_METAL)
    (void)x; (void)n; (void)w; (void)bias; (void)y;
    return -10;
#else
    (void)n;
    if (!s_dma_ready) return -6;

    uint8_t* s = (uint8_t*)scratch_buf;
    uint32_t* bias_buf = (uint32_t*)s;
    s += 128;
    uint32_t weight_words = (uint32_t)(c_in * k_h * k_w * NUM_CLUSTERS);
    uint32_t* weight_buf = (uint32_t*)s;
    s += weight_words * 4;
    uint32_t act_words_per_run = (uint32_t)((int32_t)k_h * (int32_t)CONV_ACC_MAX_W_LINE);
    uint32_t* act_buf = (uint32_t*)s;
    s += act_words_per_run * 4;
    uint32_t* out_buf = (uint32_t*)s;

    uint32_t kernel_size_u = (uint32_t)k_h;
    uint16_t img_width_u = (uint16_t)padded_w;

    for (int32_t oc_block = 0; oc_block < (c_out + NUM_PE - 1) / NUM_PE; oc_block++) {
        int32_t oc0 = oc_block * NUM_PE;
        int32_t n_oc = (oc0 + NUM_PE <= c_out) ? NUM_PE : (c_out - oc0);

        for (int i = 0; i < NUM_PE; i++)
            bias_buf[i] = (uint32_t)(i < n_oc && bias ? bias[oc0 + i] : 0);

        conv_acc_weight_repack(w, c_out, c_in, k_h, k_w, oc_block, weight_buf);

        for (int32_t row = 0; row < h_out; row++) {
            for (int32_t L = 0; L < k_h; L++) {
                int32_t line_idx = (row * stride_h) + L;
                pack_one_line_padded_to_maxw(x, c_in, padded_w, line_idx, pad_h, pad_w, h_in, w_in,
                    act_buf + (uint32_t)L * CONV_ACC_MAX_W_LINE);
            }

            uint32_t act_start_u = 0u;
            int r = conv_acc_run_once(
                kernel_size_u, (uint32_t)c_in, img_width_u, (uint32_t)stride_w, act_start_u, multiplier,
                bias_buf, weight_buf, weight_words, act_buf, act_words_per_run,
                (uint32_t)w_out, out_buf, (row == 0) ? 1 : 0);
            if (r != 0) return r;

            unpack_out_one_row(out_buf, w_out, row, oc0, c_out, y, h_out);
        }
    }
    return 0;
#endif
}
