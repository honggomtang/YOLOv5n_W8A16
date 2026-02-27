#ifndef CONV_ACC_DRIVER_H
#define CONV_ACC_DRIVER_H

#include <stdint.h>

#ifndef CONV_ACC_GPIO_BASE_0
#define CONV_ACC_GPIO_BASE_0  0x40000000u
#endif
#ifndef CONV_ACC_GPIO_BASE_1
#define CONV_ACC_GPIO_BASE_1  0x40010000u
#endif
#define CONV_ACC_CH1_ADDR     (CONV_ACC_GPIO_BASE_1 + 0x8u)
#define CONV_ACC_CH1_START_LOAD_MASK   0x1u
#define CONV_ACC_CH1_START_LOAD_SHIFT  0
#define CONV_ACC_CH1_KERNEL_SIZE_MASK  0x1Eu
#define CONV_ACC_CH1_KERNEL_SIZE_SHIFT 1
#define CONV_ACC_CH1_IMG_WIDTH_MASK    0x1FFE0u
#define CONV_ACC_CH1_IMG_WIDTH_SHIFT   5
#define CONV_ACC_CH1_STRIDE_MASK       0x1E00000u
#define CONV_ACC_CH1_STRIDE_SHIFT      21
#define CONV_ACC_CH1_ACT_START_MASK    0xFE000000u
#define CONV_ACC_CH1_ACT_START_SHIFT   25
#define CONV_ACC_CH2_ADDR     (CONV_ACC_GPIO_BASE_1 + 0x0u)

#define CONV_ACC_NUM_CLUSTERS  8
#define CONV_ACC_NUM_PE        32
#define CONV_ACC_MAX_W_LINE    3072

void conv_acc_weight_repack(
    const int8_t* w_oc_ic_kh_kw,
    int32_t oc_total, int32_t ic, int32_t kh, int32_t kw,
    int32_t oc_block_index,
    uint32_t* out);

void conv_acc_bias_quant(
    const float* bias_float,
    float scale_weight,
    int32_t c_out,
    int32_t* out);

void conv_acc_pack_activation_line(
    const int16_t* src,
    int32_t img_width,
    int32_t target_ic,
    int32_t line_idx,
    uint32_t* dst);

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
    int first_row);

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
    uint32_t scratch_size);

uint32_t conv_acc_scratch_size(int32_t c_in, int32_t k_h, int32_t k_w,
    int32_t padded_h, int32_t padded_w, int32_t h_out, int32_t w_out);

#if defined(BARE_METAL)
int conv_acc_dma_init(void);
#endif

#endif /* CONV_ACC_DRIVER_H */
