#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#ifndef BARE_METAL
#include <stdio.h>
#endif

#include "utils/weights_loader.h"
#include "utils/image_loader.h"
#ifndef USE_W8A16
#include "blocks/conv_w8a32.h"
#include "blocks/c3_w8a32.h"
#include "blocks/sppf_w8a32.h"
#include "blocks/detect_w8a32.h"
#include "operations/upsample_w8a32.h"
#include "operations/concat_w8a32.h"
#endif
#include "blocks/decode.h"
#include "blocks/nms.h"
#include "utils/feature_pool.h"
#include "utils/mcycle.h"
#include "utils/timing.h"

#ifdef USE_W8A16
#include "blocks/conv_w8a16.h"
static inline uint32_t scale_to_mult(float s) {
    if (s <= 0.f) return 1U;
    uint32_t u = (uint32_t)(s * 65536.0f + 0.5f);
    return (u < 1) ? 1U : u;
}
#include "blocks/c3_w8a16.h"
#include "blocks/sppf_w8a16.h"
#include "blocks/detect_w8a16.h"
#include "operations/upsample_w8a16.h"
#include "operations/concat_w8a16.h"
#endif
#ifdef BARE_METAL
#include "platform_config.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "utils/uart_dump.h"
#ifndef CPU_MHZ
#define CPU_MHZ 100
#endif
#define LAYER_MS(c) ((double)(c)/((double)CPU_MHZ*1000.0))
#define LAYER_MS_INT(c) ((unsigned long long)((c) / ((uint64_t)CPU_MHZ * 1000ULL)))
#define LAYER_LOG_REF(i, cycles, ptr) YOLO_LOG("  L%d %llu ms (ref: 0x%08X)\n", (i), LAYER_MS_INT(cycles), (unsigned)(*(const uint32_t*)(ptr)))
#define LAYER_LOG_VAL(i, cycles, ptr) do { \
    int16_t _vi = *(const int16_t*)(ptr); \
    float _fv = (float)_vi / 1024.0f; \
    uint32_t _fpu; memcpy(&_fpu, &_fv, sizeof(uint32_t)); \
    YOLO_LOG("  L%d %llu ms (0x%04X int) (0x%08X fp) (ref: 0x%08X)\n", (i), LAYER_MS_INT(cycles), (unsigned)((uint16_t)_vi), (unsigned)_fpu, (unsigned)LAYER_REF_HEX[i]); \
} while(0)
#else
#define LAYER_MS(c) ((c)/1000.0)
#define LAYER_LOG_REF(i, cycles, ptr) YOLO_LOG("  L%d %.2f ms (ref: 0x%08X)\n", (i), LAYER_MS(cycles), (unsigned)(*(const uint32_t*)(ptr)))
#define LAYER_LOG_VAL(i, cycles, ptr) do { \
    int16_t _vi = *(const int16_t*)(ptr); \
    float _fv = (float)_vi / 1024.0f; \
    uint32_t _fpu; memcpy(&_fpu, &_fv, sizeof(uint32_t)); \
    YOLO_LOG("  L%d %.2f ms (0x%04X int) (0x%08X fp) (ref: 0x%08X)\n", (i), LAYER_MS(cycles), (unsigned)((uint16_t)_vi), (unsigned)_fpu, (unsigned)LAYER_REF_HEX[i]); \
} while(0)
#endif

#define W(name) weights_get_tensor_data(&weights, name)
#define W_CONV(name, scale_ptr, is8_ptr) weights_get_tensor_for_conv(&weights, (name), (scale_ptr), (is8_ptr))

#define INPUT_SIZE 640
#define NUM_CLASSES 80
#define DETECT_C_OUT ((NUM_CLASSES + 5) * 3)
#define CONF_THRESHOLD 0.20f
#define IOU_THRESHOLD 0.45f
#define MAX_DETECTIONS 300

#ifndef YOLO_VERBOSE
#define YOLO_VERBOSE 1
#endif

/* 디버그 출력(기본 OFF). 필요 시 빌드 옵션으로 -DYOLO_DEBUG=1 */
#ifndef YOLO_DEBUG
#define YOLO_DEBUG 0
#endif

#if defined(BARE_METAL)
#define YOLO_LOG(...) xil_printf(__VA_ARGS__)
#elif YOLO_VERBOSE
#define YOLO_LOG(...) printf(__VA_ARGS__)
#else
#define YOLO_LOG(...) ((void)0)
#endif

static const float STRIDES[3] = {8.0f, 16.0f, 32.0f};
static const float ANCHORS[3][6] = {
    {10.0f, 13.0f, 16.0f, 30.0f, 33.0f, 23.0f},
    {30.0f, 61.0f, 62.0f, 45.0f, 59.0f, 119.0f},
    {116.0f, 90.0f, 156.0f, 198.0f, 373.0f, 326.0f}
};

static const char* const COCO_NAMES[NUM_CLASSES] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

#define Q6_10_SCALE 1024

#ifdef USE_W8A16
static const uint32_t LAYER_REF_HEX[24] = {
    0x40A99DBF, 0xBC79BD52, 0x3E15527D, 0xBE299CE3, 0xBE3D9DFD, 0xBB37A2C8,
    0x3E2BA6FF, 0xBE8BEABD, 0xBE21D650, 0xBE6A42BF, 0x3F0EE4CA, 0x3F0EE4CA,
    0x3F0EE4CA, 0xBE8E43B8, 0x3E4BD0B2, 0x3E4BD0B2, 0x3E4BD0B2, 0xBE39CAD9,
    0xBE4EEFC5, 0xBE4EEFC5, 0x3EB7151A, 0xBD6EE56C, 0xBD6EE56C, 0x3F1D5A82
};
static void w8a16_bias_convert(const float* b, float scale, int c_out, int32_t* out) {
    if (!b || scale <= 0.f) { for (int k = 0; k < c_out; k++) out[k] = 0; return; }
    float factor = 1024.0f / scale;
    for (int k = 0; k < c_out; k++)
        out[k] = (int32_t)roundf(b[k] * factor);
}

static int yolov5n_inference_w8a16(
    const preprocessed_image_t* img,
    weights_loader_t* weights,
    float* p3_out, float* p4_out, float* p5_out,
    uint64_t* out_cycles_backbone, uint64_t* out_cycles_neck, uint64_t* out_cycles_head,
    int16_t* x0_a16_zero_copy)
{
#define W_W16(name) weights_get_tensor_data(weights, name)
#define W_CONV_W16(name, s, i) weights_get_tensor_for_conv(weights, name, s, i)

    feature_pool_scratch_reset();

    const int n = 1;
    uint64_t t_stage_start, t_layer;
    uint64_t layer_cycles[25];
    uint64_t cy_backbone = 0, cy_neck = 0, cy_head = 0;

    YOLO_LOG("Backbone: ");
    t_stage_start = timer_read64();
    static int32_t bias_buf[256];

    /* L0 입력: zero-copy a16 포인터가 있으면 그대로 사용, 없으면 float 이미지를 Q6.10으로 변환 */
    const int in_elems = 1 * 3 * 640 * 640;
    int16_t* x0;
    if (x0_a16_zero_copy) {
        x0 = x0_a16_zero_copy;
    } else {
        x0 = (int16_t*)feature_pool_scratch_alloc((size_t)in_elems * sizeof(int16_t));
        if (!x0) { YOLO_LOG("ERROR: W8A16 scratch alloc input failed\n"); return 1; }
        for (int i = 0; i < in_elems; i++) {
            float v = img->data[i] * (float)Q6_10_SCALE;
            if (v > 32767.f) v = 32767.f;
            if (v < -32768.f) v = -32768.f;
            x0[i] = (int16_t)(int32_t)v;
        }
    }

    /* L0: Conv 6x6 s2 */
    size_t sz_l0 = (size_t)(1 * 16 * 320 * 320 * sizeof(int16_t));
    int16_t* l0 = (int16_t*)feature_pool_scratch_alloc(sz_l0);
    if (!l0) { YOLO_LOG("ERROR: W8A16 scratch l0 failed\n"); return 1; }
    yolo_timing_set_layer(0);
    t_layer = timer_read64();
    { float s; int i8; void* w = W_CONV_W16("model.0.conv.weight", &s, &i8);
      const float* b = (const float*)W_W16("model.0.conv.bias");
      w8a16_bias_convert(b, s, 16, bias_buf);
      conv_block_nchw_w8a16(x0, n, 3, 640, 640, (const int8_t*)w, 16, 6, 6, bias_buf, scale_to_mult(s), 2, 2, 2, 2, l0, 320, 320); }
    layer_cycles[0] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(0, layer_cycles[0], l0);
    yolo_timing_print_layer_ops(0);

    /* L1: Conv 3x3 s2 */
    size_t sz_l1 = (size_t)(1 * 32 * 160 * 160 * sizeof(int16_t));
    int16_t* l1 = (int16_t*)feature_pool_scratch_alloc(sz_l1);
    if (!l1) { YOLO_LOG("ERROR: W8A16 scratch l1 failed\n"); return 1; }
    yolo_timing_set_layer(1);
    t_layer = timer_read64();
    { float s; int i8; void* w = W_CONV_W16("model.1.conv.weight", &s, &i8);
      const float* b = (const float*)W_W16("model.1.conv.bias");
      w8a16_bias_convert(b, s, 32, bias_buf);
      conv_block_nchw_w8a16(l0, n, 16, 320, 320, (const int8_t*)w, 32, 3, 3, bias_buf, scale_to_mult(s), 2, 2, 1, 1, l1, 160, 160); }
    layer_cycles[1] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(1, layer_cycles[1], l1);
    yolo_timing_print_layer_ops(1);

    /* L2: C3 n=1 */
    size_t sz_l2 = (size_t)(1 * 32 * 160 * 160 * sizeof(int16_t));
    int16_t* l2 = (int16_t*)feature_pool_scratch_alloc(sz_l2);
    if (!l2) { YOLO_LOG("ERROR: W8A16 scratch l2 failed\n"); return 1; }
    yolo_timing_set_layer(2);
    t_layer = timer_read64();
    { const char* bn_cv1_n[1] = { "model.2.m.0.cv1.conv.weight" };
      const char* bn_cv2_n[1] = { "model.2.m.0.cv2.conv.weight" };
      c3_nchw_w8a16(weights, l1, n, 32, 160, 160,
          "model.2.cv1.conv.weight", "model.2.cv2.conv.weight", "model.2.cv3.conv.weight",
          1, bn_cv1_n, bn_cv2_n, 1, l2); }
    layer_cycles[2] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(2, layer_cycles[2], l2);
    yolo_timing_print_layer_ops(2);

    /* L3: Conv 3x3 s2 */
    size_t sz_l3 = (size_t)(1 * 64 * 80 * 80 * sizeof(int16_t));
    int16_t* l3 = (int16_t*)feature_pool_scratch_alloc(sz_l3);
    if (!l3) { YOLO_LOG("ERROR: W8A16 scratch l3 failed\n"); return 1; }
    yolo_timing_set_layer(3);
    t_layer = timer_read64();
    { float s; int i8; void* w = W_CONV_W16("model.3.conv.weight", &s, &i8);
      const float* b = (const float*)W_W16("model.3.conv.bias");
      w8a16_bias_convert(b, s, 64, bias_buf);
      conv_block_nchw_w8a16(l2, n, 32, 160, 160, (const int8_t*)w, 64, 3, 3, bias_buf, scale_to_mult(s), 2, 2, 1, 1, l3, 80, 80); }
    layer_cycles[3] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(3, layer_cycles[3], l3);
    yolo_timing_print_layer_ops(3);

    /* L4: C3 n=2 */
    size_t sz_l4 = (size_t)(1 * 64 * 80 * 80 * sizeof(int16_t));
    int16_t* l4 = (int16_t*)feature_pool_scratch_alloc(sz_l4);
    if (!l4) { YOLO_LOG("ERROR: W8A16 scratch l4 failed\n"); return 1; }
    yolo_timing_set_layer(4);
    t_layer = timer_read64();
    { const char* bn_cv1_n[2] = { "model.4.m.0.cv1.conv.weight", "model.4.m.1.cv1.conv.weight" };
      const char* bn_cv2_n[2] = { "model.4.m.0.cv2.conv.weight", "model.4.m.1.cv2.conv.weight" };
      c3_nchw_w8a16(weights, l3, n, 64, 80, 80,
          "model.4.cv1.conv.weight", "model.4.cv2.conv.weight", "model.4.cv3.conv.weight",
          2, bn_cv1_n, bn_cv2_n, 1, l4); }
    layer_cycles[4] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(4, layer_cycles[4], l4);
    yolo_timing_print_layer_ops(4);

    /* L5: Conv 3x3 s2 */
    size_t sz_l5 = (size_t)(1 * 128 * 40 * 40 * sizeof(int16_t));
    int16_t* l5 = (int16_t*)feature_pool_scratch_alloc(sz_l5);
    if (!l5) { YOLO_LOG("ERROR: W8A16 scratch l5 failed\n"); return 1; }
    yolo_timing_set_layer(5);
    t_layer = timer_read64();
    { float s; int i8; void* w = W_CONV_W16("model.5.conv.weight", &s, &i8);
      const float* b = (const float*)W_W16("model.5.conv.bias");
      w8a16_bias_convert(b, s, 128, bias_buf);
      conv_block_nchw_w8a16(l4, n, 64, 80, 80, (const int8_t*)w, 128, 3, 3, bias_buf, scale_to_mult(s), 2, 2, 1, 1, l5, 40, 40); }
    layer_cycles[5] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(5, layer_cycles[5], l5);
    yolo_timing_print_layer_ops(5);

    /* L6: C3 n=3 */
    size_t sz_l6 = (size_t)(1 * 128 * 40 * 40 * sizeof(int16_t));
    int16_t* l6 = (int16_t*)feature_pool_scratch_alloc(sz_l6);
    if (!l6) { YOLO_LOG("ERROR: W8A16 scratch l6 failed\n"); return 1; }
    yolo_timing_set_layer(6);
    t_layer = timer_read64();
    { const char* bn_cv1_n[3] = { "model.6.m.0.cv1.conv.weight", "model.6.m.1.cv1.conv.weight", "model.6.m.2.cv1.conv.weight" };
      const char* bn_cv2_n[3] = { "model.6.m.0.cv2.conv.weight", "model.6.m.1.cv2.conv.weight", "model.6.m.2.cv2.conv.weight" };
      c3_nchw_w8a16(weights, l5, n, 128, 40, 40,
          "model.6.cv1.conv.weight", "model.6.cv2.conv.weight", "model.6.cv3.conv.weight",
          3, bn_cv1_n, bn_cv2_n, 1, l6); }
    layer_cycles[6] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(6, layer_cycles[6], l6);
    yolo_timing_print_layer_ops(6);

    /* L7: Conv 3x3 s2 */
    size_t sz_l7 = (size_t)(1 * 256 * 20 * 20 * sizeof(int16_t));
    int16_t* l7 = (int16_t*)feature_pool_scratch_alloc(sz_l7);
    if (!l7) { YOLO_LOG("ERROR: W8A16 scratch l7 failed\n"); return 1; }
    yolo_timing_set_layer(7);
    t_layer = timer_read64();
    { float s; int i8; void* w = W_CONV_W16("model.7.conv.weight", &s, &i8);
      const float* b = (const float*)W_W16("model.7.conv.bias");
      w8a16_bias_convert(b, s, 256, bias_buf);
      conv_block_nchw_w8a16(l6, n, 128, 40, 40, (const int8_t*)w, 256, 3, 3, bias_buf, scale_to_mult(s), 2, 2, 1, 1, l7, 20, 20); }
    layer_cycles[7] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(7, layer_cycles[7], l7);
    yolo_timing_print_layer_ops(7);

    /* L8: C3 n=1 */
    size_t sz_l8 = (size_t)(1 * 256 * 20 * 20 * sizeof(int16_t));
    int16_t* l8 = (int16_t*)feature_pool_scratch_alloc(sz_l8);
    if (!l8) { YOLO_LOG("ERROR: W8A16 scratch l8 failed\n"); return 1; }
    yolo_timing_set_layer(8);
    t_layer = timer_read64();
    { const char* bn_cv1_n[1] = { "model.8.m.0.cv1.conv.weight" };
      const char* bn_cv2_n[1] = { "model.8.m.0.cv2.conv.weight" };
      c3_nchw_w8a16(weights, l7, n, 256, 20, 20,
          "model.8.cv1.conv.weight", "model.8.cv2.conv.weight", "model.8.cv3.conv.weight",
          1, bn_cv1_n, bn_cv2_n, 1, l8); }
    layer_cycles[8] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(8, layer_cycles[8], l8);
    yolo_timing_print_layer_ops(8);

    /* L9: SPPF */
    size_t sz_l9 = (size_t)(1 * 256 * 20 * 20 * sizeof(int16_t));
    int16_t* l9 = (int16_t*)feature_pool_scratch_alloc(sz_l9);
    if (!l9) { YOLO_LOG("ERROR: W8A16 scratch l9 failed\n"); return 1; }
    yolo_timing_set_layer(9);
    t_layer = timer_read64();
    sppf_nchw_w8a16(weights, l8, n, 256, 20, 20, "model.9.cv1.conv.weight", "model.9.cv2.conv.weight", 5, l9);
    layer_cycles[9] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(9, layer_cycles[9], l9);
    yolo_timing_print_layer_ops(9);
    cy_backbone = timer_delta64(t_stage_start, timer_read64());
    YOLO_LOG("\nNeck: ");
    t_stage_start = timer_read64();

    /* L10: Conv 1x1 */
    size_t sz_l10 = (size_t)(1 * 128 * 20 * 20 * sizeof(int16_t));
    int16_t* l10 = (int16_t*)feature_pool_scratch_alloc(sz_l10);
    if (!l10) { YOLO_LOG("ERROR: W8A16 scratch l10 failed\n"); return 1; }
    yolo_timing_set_layer(10);
    t_layer = timer_read64();
    { float s; int i8; void* w = W_CONV_W16("model.10.conv.weight", &s, &i8);
      const float* b = (const float*)W_W16("model.10.conv.bias");
      w8a16_bias_convert(b, s, 128, bias_buf);
      conv_block_nchw_w8a16(l9, n, 256, 20, 20, (const int8_t*)w, 128, 1, 1, bias_buf, scale_to_mult(s), 1, 1, 0, 0, l10, 20, 20); }
    layer_cycles[10] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(10, layer_cycles[10], l10);
    yolo_timing_print_layer_ops(10);

    /* L11: Upsample */
    size_t sz_l11 = (size_t)(1 * 128 * 40 * 40 * sizeof(int16_t));
    int16_t* l11 = (int16_t*)feature_pool_scratch_alloc(sz_l11);
    if (!l11) { YOLO_LOG("ERROR: W8A16 scratch l11 failed\n"); return 1; }
    yolo_timing_set_layer(11);
    t_layer = timer_read64();
    upsample_nearest2x_nchw_w8a16(l10, n, 128, 20, 20, l11);
    layer_cycles[11] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(11, layer_cycles[11], l11);
    yolo_timing_print_layer_ops(11);

    /* L12: Concat l11 + l6 */
    size_t sz_l12 = (size_t)(1 * 256 * 40 * 40 * sizeof(int16_t));
    int16_t* l12 = (int16_t*)feature_pool_scratch_alloc(sz_l12);
    if (!l12) { YOLO_LOG("ERROR: W8A16 scratch l12 failed\n"); return 1; }
    yolo_timing_set_layer(12);
    t_layer = timer_read64();
    yolo_timing_begin("concat");
    concat_nchw_w8a16(l11, 128, l6, 128, n, 40, 40, l12);
    yolo_timing_end();
    layer_cycles[12] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(12, layer_cycles[12], l12);
    yolo_timing_print_layer_ops(12);

    /* L13: C3 n=1, shortcut=0 */
    size_t sz_l13 = (size_t)(1 * 128 * 40 * 40 * sizeof(int16_t));
    int16_t* l13 = (int16_t*)feature_pool_scratch_alloc(sz_l13);
    if (!l13) { YOLO_LOG("ERROR: W8A16 scratch l13 failed\n"); return 1; }
    yolo_timing_set_layer(13);
    t_layer = timer_read64();
    { const char* bn_cv1_n[1] = { "model.13.m.0.cv1.conv.weight" };
      const char* bn_cv2_n[1] = { "model.13.m.0.cv2.conv.weight" };
      c3_nchw_w8a16(weights, l12, n, 256, 40, 40,
          "model.13.cv1.conv.weight", "model.13.cv2.conv.weight", "model.13.cv3.conv.weight",
          1, bn_cv1_n, bn_cv2_n, 0, l13); }
    layer_cycles[13] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(13, layer_cycles[13], l13);
    yolo_timing_print_layer_ops(13);

    /* L14: Conv 1x1 */
    size_t sz_l14 = (size_t)(1 * 64 * 40 * 40 * sizeof(int16_t));
    int16_t* l14 = (int16_t*)feature_pool_scratch_alloc(sz_l14);
    if (!l14) { YOLO_LOG("ERROR: W8A16 scratch l14 failed\n"); return 1; }
    yolo_timing_set_layer(14);
    t_layer = timer_read64();
    { float s; int i8; void* w = W_CONV_W16("model.14.conv.weight", &s, &i8);
      const float* b = (const float*)W_W16("model.14.conv.bias");
      w8a16_bias_convert(b, s, 64, bias_buf);
      conv_block_nchw_w8a16(l13, n, 128, 40, 40, (const int8_t*)w, 64, 1, 1, bias_buf, scale_to_mult(s), 1, 1, 0, 0, l14, 40, 40); }
    layer_cycles[14] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(14, layer_cycles[14], l14);
    yolo_timing_print_layer_ops(14);

    /* L15: Upsample */
    size_t sz_l15 = (size_t)(1 * 64 * 80 * 80 * sizeof(int16_t));
    int16_t* l15 = (int16_t*)feature_pool_scratch_alloc(sz_l15);
    if (!l15) { YOLO_LOG("ERROR: W8A16 scratch l15 failed\n"); return 1; }
    yolo_timing_set_layer(15);
    t_layer = timer_read64();
    upsample_nearest2x_nchw_w8a16(l14, n, 64, 40, 40, l15);
    layer_cycles[15] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(15, layer_cycles[15], l15);
    yolo_timing_print_layer_ops(15);

    /* L16: Concat l15 + l4 */
    size_t sz_l16 = (size_t)(1 * 128 * 80 * 80 * sizeof(int16_t));
    int16_t* l16 = (int16_t*)feature_pool_scratch_alloc(sz_l16);
    if (!l16) { YOLO_LOG("ERROR: W8A16 scratch l16 failed\n"); return 1; }
    yolo_timing_set_layer(16);
    t_layer = timer_read64();
    yolo_timing_begin("concat");
    concat_nchw_w8a16(l15, 64, l4, 64, n, 80, 80, l16);
    yolo_timing_end();
    layer_cycles[16] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(16, layer_cycles[16], l16);
    yolo_timing_print_layer_ops(16);

    /* L17: C3 n=1, shortcut=0 -> P3 */
    size_t sz_l17 = (size_t)(1 * 64 * 80 * 80 * sizeof(int16_t));
    int16_t* l17 = (int16_t*)feature_pool_scratch_alloc(sz_l17);
    if (!l17) { YOLO_LOG("ERROR: W8A16 scratch l17 failed\n"); return 1; }
    yolo_timing_set_layer(17);
    t_layer = timer_read64();
    { const char* bn_cv1_n[1] = { "model.17.m.0.cv1.conv.weight" };
      const char* bn_cv2_n[1] = { "model.17.m.0.cv2.conv.weight" };
      c3_nchw_w8a16(weights, l16, n, 128, 80, 80,
          "model.17.cv1.conv.weight", "model.17.cv2.conv.weight", "model.17.cv3.conv.weight",
          1, bn_cv1_n, bn_cv2_n, 0, l17); }
    layer_cycles[17] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(17, layer_cycles[17], l17);
    yolo_timing_print_layer_ops(17);

    /* L18: Conv 3x3 s2 */
    size_t sz_l18 = (size_t)(1 * 64 * 40 * 40 * sizeof(int16_t));
    int16_t* l18 = (int16_t*)feature_pool_scratch_alloc(sz_l18);
    if (!l18) { YOLO_LOG("ERROR: W8A16 scratch l18 failed\n"); return 1; }
    yolo_timing_set_layer(18);
    t_layer = timer_read64();
    { float s; int i8; void* w = W_CONV_W16("model.18.conv.weight", &s, &i8);
      const float* b = (const float*)W_W16("model.18.conv.bias");
      w8a16_bias_convert(b, s, 64, bias_buf);
      conv_block_nchw_w8a16(l17, n, 64, 80, 80, (const int8_t*)w, 64, 3, 3, bias_buf, scale_to_mult(s), 2, 2, 1, 1, l18, 40, 40); }
    layer_cycles[18] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(18, layer_cycles[18], l18);
    yolo_timing_print_layer_ops(18);

    /* L19: Concat l18 + l14 */
    size_t sz_l19 = (size_t)(1 * 128 * 40 * 40 * sizeof(int16_t));
    int16_t* l19 = (int16_t*)feature_pool_scratch_alloc(sz_l19);
    if (!l19) { YOLO_LOG("ERROR: W8A16 scratch l19 failed\n"); return 1; }
    yolo_timing_set_layer(19);
    t_layer = timer_read64();
    yolo_timing_begin("concat");
    concat_nchw_w8a16(l18, 64, l14, 64, n, 40, 40, l19);
    yolo_timing_end();
    layer_cycles[19] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(19, layer_cycles[19], l19);
    yolo_timing_print_layer_ops(19);

    /* L20: C3 n=1, shortcut=0 -> P4 */
    size_t sz_l20 = (size_t)(1 * 128 * 40 * 40 * sizeof(int16_t));
    int16_t* l20 = (int16_t*)feature_pool_scratch_alloc(sz_l20);
    if (!l20) { YOLO_LOG("ERROR: W8A16 scratch l20 failed\n"); return 1; }
    yolo_timing_set_layer(20);
    t_layer = timer_read64();
    { const char* bn_cv1_n[1] = { "model.20.m.0.cv1.conv.weight" };
      const char* bn_cv2_n[1] = { "model.20.m.0.cv2.conv.weight" };
      c3_nchw_w8a16(weights, l19, n, 128, 40, 40,
          "model.20.cv1.conv.weight", "model.20.cv2.conv.weight", "model.20.cv3.conv.weight",
          1, bn_cv1_n, bn_cv2_n, 0, l20); }
    layer_cycles[20] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(20, layer_cycles[20], l20);
    yolo_timing_print_layer_ops(20);

    /* L21: Conv 3x3 s2 */
    size_t sz_l21 = (size_t)(1 * 128 * 20 * 20 * sizeof(int16_t));
    int16_t* l21 = (int16_t*)feature_pool_scratch_alloc(sz_l21);
    if (!l21) { YOLO_LOG("ERROR: W8A16 scratch l21 failed\n"); return 1; }
    yolo_timing_set_layer(21);
    t_layer = timer_read64();
    { float s; int i8; void* w = W_CONV_W16("model.21.conv.weight", &s, &i8);
      const float* b = (const float*)W_W16("model.21.conv.bias");
      w8a16_bias_convert(b, s, 128, bias_buf);
      conv_block_nchw_w8a16(l20, n, 128, 40, 40, (const int8_t*)w, 128, 3, 3, bias_buf, scale_to_mult(s), 2, 2, 1, 1, l21, 20, 20); }
    layer_cycles[21] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(21, layer_cycles[21], l21);
    yolo_timing_print_layer_ops(21);

    /* L22: Concat l21 + l10 */
    size_t sz_l22 = (size_t)(1 * 256 * 20 * 20 * sizeof(int16_t));
    int16_t* l22 = (int16_t*)feature_pool_scratch_alloc(sz_l22);
    if (!l22) { YOLO_LOG("ERROR: W8A16 scratch l22 failed\n"); return 1; }
    yolo_timing_set_layer(22);
    t_layer = timer_read64();
    yolo_timing_begin("concat");
    concat_nchw_w8a16(l21, 128, l10, 128, n, 20, 20, l22);
    yolo_timing_end();
    layer_cycles[22] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(22, layer_cycles[22], l22);
    yolo_timing_print_layer_ops(22);

    /* L23: C3 n=1, shortcut=0 -> P5 */
    size_t sz_l23 = (size_t)(1 * 256 * 20 * 20 * sizeof(int16_t));
    int16_t* l23 = (int16_t*)feature_pool_scratch_alloc(sz_l23);
    if (!l23) { YOLO_LOG("ERROR: W8A16 scratch l23 failed\n"); return 1; }
    yolo_timing_set_layer(23);
    t_layer = timer_read64();
    { const char* bn_cv1_n[1] = { "model.23.m.0.cv1.conv.weight" };
      const char* bn_cv2_n[1] = { "model.23.m.0.cv2.conv.weight" };
      c3_nchw_w8a16(weights, l22, n, 256, 20, 20,
          "model.23.cv1.conv.weight", "model.23.cv2.conv.weight", "model.23.cv3.conv.weight",
          1, bn_cv1_n, bn_cv2_n, 0, l23); }
    layer_cycles[23] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_VAL(23, layer_cycles[23], l23);
    yolo_timing_print_layer_ops(23);
    cy_neck = timer_delta64(t_stage_start, timer_read64());
    YOLO_LOG("\nHead: ");
    t_stage_start = timer_read64();

    /* Detect: 3x 1x1 Conv */
    const int elems_p3 = 1 * DETECT_C_OUT * 80 * 80;
    const int elems_p4 = 1 * DETECT_C_OUT * 40 * 40;
    const int elems_p5 = 1 * DETECT_C_OUT * 20 * 20;
    int16_t* p3_i16 = (int16_t*)feature_pool_scratch_alloc((size_t)elems_p3 * sizeof(int16_t));
    int16_t* p4_i16 = (int16_t*)feature_pool_scratch_alloc((size_t)elems_p4 * sizeof(int16_t));
    int16_t* p5_i16 = (int16_t*)feature_pool_scratch_alloc((size_t)elems_p5 * sizeof(int16_t));
    if (!p3_i16 || !p4_i16 || !p5_i16) { YOLO_LOG("ERROR: W8A16 scratch detect out failed\n"); return 1; }
    yolo_timing_set_layer(24);
    detect_nchw_w8a16(weights, l17, 64, 80, 80, l20, 128, 40, 40, l23, 256, 20, 20,
        "model.24.m.0.weight", "model.24.m.1.weight", "model.24.m.2.weight",
        DETECT_C_OUT, p3_i16, p4_i16, p5_i16);
    /* Q6.10 → float: /1024 */
    for (int i = 0; i < elems_p3; i++) p3_out[i] = (float)p3_i16[i] / 1024.0f;
    for (int i = 0; i < elems_p4; i++) p4_out[i] = (float)p4_i16[i] / 1024.0f;
    for (int i = 0; i < elems_p5; i++) p5_out[i] = (float)p5_i16[i] / 1024.0f;
    cy_head = timer_delta64(t_stage_start, timer_read64());
    YOLO_LOG("Detect\n");
#ifdef BARE_METAL
    YOLO_LOG("  det %llu ms\n", LAYER_MS_INT(cy_head));
#else
    YOLO_LOG("  det %.2f ms\n", LAYER_MS(cy_head));
#endif
    yolo_timing_print_layer_ops(24);

    if (out_cycles_backbone) *out_cycles_backbone = cy_backbone;
    if (out_cycles_neck) *out_cycles_neck = cy_neck;
    if (out_cycles_head) *out_cycles_head = cy_head;

#undef W_W16
#undef W_CONV_W16
    return 0;
}
#endif /* USE_W8A16 */

int main(int argc, char* argv[]) {
#if defined(BARE_METAL)
    (void)argc;
    (void)argv;
#endif
#ifndef BARE_METAL
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
#endif

    YOLO_LOG("=== YOLOv5n Inference (Fused) ===\n\n");
    
    preprocessed_image_t img;
    weights_loader_t weights;
#ifdef USE_W8A16
#ifndef BARE_METAL
    void* a16_file_buf = NULL;
    int16_t* x0_a16_ptr = NULL;
#endif
#endif

#ifdef BARE_METAL
    Xil_DCacheInvalidateRange((uintptr_t)WEIGHTS_DDR_BASE, (unsigned int)WEIGHTS_DDR_SIZE);
#ifdef USE_W8A16
    Xil_DCacheInvalidateRange((uintptr_t)IMAGE_DDR_BASE, (unsigned int)IMAGE_A16_DDR_SIZE);
#else
    Xil_DCacheInvalidateRange((uintptr_t)IMAGE_DDR_BASE, (unsigned int)IMAGE_DDR_SIZE);
#endif
    Xil_DCacheInvalidateRange((uintptr_t)FEATURE_POOL_BASE, (unsigned int)FEATURE_POOL_SIZE);
    Xil_DCacheInvalidateRange((uintptr_t)DETECT_HEAD_BASE, (unsigned int)DETECT_HEAD_SIZE);
    Xil_DCacheEnable();

#ifdef USE_W8A16
    /* W8A16: preprocessed_image_a16.bin (24B 헤더 + int16) zero-copy. 헤더만 파싱. */
    YOLO_LOG("Loading image (a16) from DDR 0x%08X...\n", (unsigned int)IMAGE_DDR_BASE);
    if (image_init_from_memory_a16((uintptr_t)IMAGE_DDR_BASE, (size_t)IMAGE_A16_DDR_SIZE, &img) != 0) {
        YOLO_LOG("ERROR: Failed to load image (a16) header from DDR\n");
        return 1;
    }
#else
    YOLO_LOG("Loading image from DDR 0x%08X...\n", (unsigned int)IMAGE_DDR_BASE);
    if (image_init_from_memory((uintptr_t)IMAGE_DDR_BASE, (size_t)IMAGE_DDR_SIZE, &img) != 0) {
        YOLO_LOG("ERROR: Failed to load image from DDR\n");
        return 1;
    }
    img.data = (float*)((uintptr_t)IMAGE_DDR_BASE + (uintptr_t)IMAGE_HEADER_SIZE);
#endif
#ifdef USE_WEIGHTS_W8
    YOLO_LOG("Loading weights (W8) from DDR 0x%08X (size %u bytes)...\n",
             (unsigned int)WEIGHTS_W8_DDR_BASE, (unsigned)WEIGHTS_W8_DDR_SIZE);
    if (weights_init_from_memory_w8((uintptr_t)WEIGHTS_W8_DDR_BASE, (size_t)WEIGHTS_W8_DDR_SIZE, &weights) != 0) {
        YOLO_LOG("ERROR: Failed to load weights (W8) from DDR\n");
#ifdef BARE_METAL
        {
            const uint32_t* first = (const uint32_t*)(uintptr_t)WEIGHTS_W8_DDR_BASE;
            YOLO_LOG("  Debug: first word at 0x88000000 = 0x%08X (expected num_tensors ~121)\n", (unsigned)*first);
            YOLO_LOG("  Check: dow -data <path>/weights_w8.bin 0x88000000 before running ELF\n");
        }
#endif
        image_free(&img);
        return 1;
    }
#else
    YOLO_LOG("Loading weights from DDR 0x%08X...\n", (unsigned int)WEIGHTS_DDR_BASE);
    if (weights_init_from_memory((uintptr_t)WEIGHTS_DDR_BASE, (size_t)WEIGHTS_DDR_SIZE, &weights) != 0) {
        YOLO_LOG("ERROR: Failed to load weights from DDR\n");
        image_free(&img);
        return 1;
    }
#endif
    if (YOLO_DEBUG) {
        const float* bias24 = (const float*)W("model.24.m.0.bias");
        if (bias24) {
            uint32_t u0 = *(const uint32_t*)&bias24[0];
            uint32_t u4 = *(const uint32_t*)&bias24[4];
            YOLO_LOG("DEBUG bias24 @0x%08X [0]=0x%08X [4]=0x%08X\n",
                     (unsigned)(uintptr_t)bias24, (unsigned)u0, (unsigned)u4);
        }
    }
#else
#ifdef USE_W8A16
    if (image_load_from_bin_a16("data/input/preprocessed_image_a16.bin", &img, &a16_file_buf) != 0) {
        fprintf(stderr, "Failed to load image (a16)\n");
        return 1;
    }
    x0_a16_ptr = (int16_t*)((char*)a16_file_buf + 24);
#else
    if (image_load_from_bin("data/input/preprocessed_image.bin", &img) != 0) {
        fprintf(stderr, "Failed to load image\n");
        return 1;
    }
#endif
#ifdef USE_WEIGHTS_W8
    if (weights_load_from_file_w8("assets/weights_w8.bin", &weights) != 0) {
        fprintf(stderr, "Failed to load weights (W8)\n");
        image_free(&img);
        return 1;
    }
#else
    if (weights_load_from_file("assets/weights.bin", &weights) != 0) {
        fprintf(stderr, "Failed to load weights\n");
        image_free(&img);
        return 1;
    }
#endif
#endif
    YOLO_LOG("Image: %dx%d\n", img.w, img.h);
    YOLO_LOG("Weights: %d tensors\n\n", weights.num_tensors);

    feature_pool_init();
    const int n = 1;

    size_t sz_l0  = (size_t)(1 * 16  * 320 * 320 * sizeof(float));
    size_t sz_l1  = (size_t)(1 * 32  * 160 * 160 * sizeof(float));
    size_t sz_l2  = (size_t)(1 * 32  * 160 * 160 * sizeof(float));
    size_t sz_l3  = (size_t)(1 * 64  * 80  * 80  * sizeof(float));
    size_t sz_l4  = (size_t)(1 * 64  * 80  * 80  * sizeof(float));
    size_t sz_l5  = (size_t)(1 * 128 * 40  * 40  * sizeof(float));
    size_t sz_l6  = (size_t)(1 * 128 * 40  * 40  * sizeof(float));
    size_t sz_l7  = (size_t)(1 * 256 * 20  * 20  * sizeof(float));
    size_t sz_l8  = (size_t)(1 * 256 * 20  * 20  * sizeof(float));
    size_t sz_l9  = (size_t)(1 * 256 * 20  * 20  * sizeof(float));
    size_t sz_l10 = (size_t)(1 * 128 * 20  * 20  * sizeof(float));
    size_t sz_l11 = (size_t)(1 * 128 * 40  * 40  * sizeof(float));
    size_t sz_l12 = (size_t)(1 * 256 * 40  * 40  * sizeof(float));
    size_t sz_l13 = (size_t)(1 * 128 * 40  * 40  * sizeof(float));
    size_t sz_l14 = (size_t)(1 * 64  * 40  * 40  * sizeof(float));
    size_t sz_l15 = (size_t)(1 * 64  * 80  * 80  * sizeof(float));
    size_t sz_l16 = (size_t)(1 * 128 * 80  * 80  * sizeof(float));
    size_t sz_l17 = (size_t)(1 * 64  * 80  * 80  * sizeof(float));
    size_t sz_l18 = (size_t)(1 * 64  * 40  * 40  * sizeof(float));
    size_t sz_l19 = (size_t)(1 * 128 * 40  * 40  * sizeof(float));
    size_t sz_l20 = (size_t)(1 * 128 * 40  * 40  * sizeof(float));
    size_t sz_l21 = (size_t)(1 * 128 * 20  * 20  * sizeof(float));
    size_t sz_l22 = (size_t)(1 * 256 * 20  * 20  * sizeof(float));
    size_t sz_l23 = (size_t)(1 * 256 * 20  * 20  * sizeof(float));
    size_t sz_p3  = (size_t)(1 * DETECT_C_OUT * 80  * 80  * sizeof(float));
    size_t sz_p4  = (size_t)(1 * DETECT_C_OUT * 40  * 40  * sizeof(float));
    size_t sz_p5  = (size_t)(1 * DETECT_C_OUT * 20  * 20  * sizeof(float));

    float* l0 = NULL, * l1 = NULL, * l2 = NULL, * l3 = NULL, * l4 = NULL;
    float* l5 = NULL, * l6 = NULL, * l7 = NULL, * l8 = NULL, * l9 = NULL;
    float* l10 = NULL, * l11 = NULL, * l12 = NULL, * l13 = NULL, * l14 = NULL;
    float* l15 = NULL, * l16 = NULL, * l17 = NULL, * l18 = NULL, * l19 = NULL;
    float* l20 = NULL, * l21 = NULL, * l22 = NULL, * l23 = NULL;
    float* p3 = NULL, * p4 = NULL, * p5 = NULL;

#define POOL_ALLOC(ptr, sz) do { \
    (ptr) = (float*)feature_pool_alloc(sz); \
    if (!(ptr)) { \
        YOLO_LOG("ERROR: Feature pool allocation failed\n"); \
        feature_pool_reset(); weights_free(&weights); image_free(&img); \
        return 1; \
    } \
} while(0)

#ifdef BARE_METAL
#ifdef USE_W8A16
    Xil_DCacheInvalidateRange((uintptr_t)IMAGE_DDR_BASE, (unsigned int)IMAGE_A16_DDR_SIZE);
#else
    Xil_DCacheInvalidateRange((uintptr_t)IMAGE_DDR_BASE, (unsigned int)IMAGE_DDR_SIZE);
#endif
    Xil_DCacheInvalidateRange((uintptr_t)WEIGHTS_DDR_BASE, (unsigned int)WEIGHTS_DDR_SIZE);
    if (YOLO_DEBUG) {
        float img0 = img.data ? img.data[0] : 0.0f;
        uint32_t u_img = *(const uint32_t*)(&img0);
#ifdef USE_WEIGHTS_W8
        { float _sw; int _iw; void* _pw = W_CONV("model.0.conv.weight", &_sw, &_iw);
          uint32_t u_w = _pw && _iw ? (uint32_t)((const int8_t*)_pw)[0] : 0u;
          YOLO_LOG("DEBUG img0=0x%08X w0b0=0x%02X\n", (unsigned)u_img, (unsigned)u_w); }
#else
        { const float* pw = (const float*)W("model.0.conv.weight");
          uint32_t u_w = pw ? *(const uint32_t*)pw : 0u;
          YOLO_LOG("DEBUG img0=0x%08X w0f0=0x%08X\n", (unsigned)u_img, (unsigned)u_w); }
#endif
    }
#endif
    YOLO_LOG("Running inference...\n");
    yolo_timing_reset();
    uint64_t t_total_start = timer_read64();
    uint64_t t_stage_start;
    uint64_t t_layer;
    uint64_t cycles_backbone = 0, cycles_neck = 0, cycles_head = 0, cycles_decode = 0, cycles_nms = 0;
    uint64_t layer_cycles[24];  /* L0..L23 per-layer (op only) */

#ifdef USE_W8A16
    YOLO_LOG("W8A16 path: scratch_reset + full pipeline -> float p3/p4/p5\n");
#ifdef BARE_METAL
    /* DDR 고정 영역 사용 (힙 부족 방지). W8A32와 동일한 DETECT_HEAD_BASE 레이아웃. */
    p3 = (float*)(uintptr_t)DETECT_HEAD_BASE;
    p4 = p3 + (DETECT_C_OUT * 80 * 80);
    p5 = p4 + (DETECT_C_OUT * 40 * 40);
#else
    p3 = (float*)malloc(sz_p3);
    p4 = (float*)malloc(sz_p4);
    p5 = (float*)malloc(sz_p5);
    if (!p3 || !p4 || !p5) {
        YOLO_LOG("ERROR: W8A16 output buffer alloc failed\n");
        if (p3) free(p3); if (p4) free(p4); if (p5) free(p5);
        feature_pool_reset();
        weights_free(&weights);
        image_free(&img);
        return 1;
    }
#endif
    if (yolov5n_inference_w8a16(&img, &weights, p3, p4, p5,
            &cycles_backbone, &cycles_neck, &cycles_head,
#ifdef BARE_METAL
            (int16_t*)((uintptr_t)IMAGE_DDR_BASE + (uintptr_t)IMAGE_HEADER_SIZE)
#else
            x0_a16_ptr
#endif
    ) != 0) {
        YOLO_LOG("ERROR: W8A16 inference failed\n");
#ifndef BARE_METAL
        free(p3); free(p4); free(p5);
        if (a16_file_buf) free(a16_file_buf);
#endif
        feature_pool_reset();
        weights_free(&weights);
        image_free(&img);
        return 1;
    }
#ifndef BARE_METAL
    if (a16_file_buf) { free(a16_file_buf); a16_file_buf = NULL; }
    free(p3); free(p4); free(p5);
#endif
#else
    YOLO_LOG("Backbone: ");

    // ===== Backbone =====
    t_stage_start = timer_read64();
    yolo_timing_set_layer(0);
    // Layer 0: Conv 6x6 s2
    POOL_ALLOC(l0, sz_l0);
    t_layer = timer_read64();
    { float _sw; int _iw; void* _pw = W_CONV("model.0.conv.weight", &_sw, &_iw);
      conv_block_nchw_f32_w8a32(img.data, n, 3, 640, 640, _pw, _sw, _iw, 16, 6, 6, 2, 2, 2, 2,
          W("model.0.conv.bias"), l0, 320, 320); }
    layer_cycles[0] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(0, layer_cycles[0], &l0[0]);
    yolo_timing_print_layer_ops(0);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l0, 16);
#endif

    yolo_timing_set_layer(1);
    // Layer 1: Conv 3x3 s2
    POOL_ALLOC(l1, sz_l1);
    t_layer = timer_read64();
    { float _sw; int _iw; void* _pw = W_CONV("model.1.conv.weight", &_sw, &_iw);
      conv_block_nchw_f32_w8a32(l0, n, 16, 320, 320, _pw, _sw, _iw, 32, 3, 3, 2, 2, 1, 1,
          W("model.1.conv.bias"), l1, 160, 160); }
    layer_cycles[1] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(1, layer_cycles[1], &l1[0]);
    yolo_timing_print_layer_ops(1);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l1, 16);
#endif
    feature_pool_free(l0);

    yolo_timing_set_layer(2);
    // Layer 2: C3 (n=1)
#ifdef BARE_METAL
    { size_t largest = feature_pool_get_largest_free(); YOLO_LOG("  before L2 pool largest_free=%u\n", (unsigned)largest); }
#endif
    POOL_ALLOC(l2, sz_l2);
    float l2_cv1_scale[1]; int l2_cv1_is_int8[1]; const void* l2_cv1w[1]; l2_cv1w[0] = W_CONV("model.2.m.0.cv1.conv.weight", &l2_cv1_scale[0], &l2_cv1_is_int8[0]);
    float l2_cv2_scale[1]; int l2_cv2_is_int8[1]; const void* l2_cv2w[1]; l2_cv2w[0] = W_CONV("model.2.m.0.cv2.conv.weight", &l2_cv2_scale[0], &l2_cv2_is_int8[0]);
    const float* l2_cv1b[] = {W("model.2.m.0.cv1.conv.bias")};
    const float* l2_cv2b[] = {W("model.2.m.0.cv2.conv.bias")};
    { float s1, s2, s3; int i1, i2, i3; void* w1 = W_CONV("model.2.cv1.conv.weight", &s1, &i1); void* w2 = W_CONV("model.2.cv2.conv.weight", &s2, &i2); void* w3 = W_CONV("model.2.cv3.conv.weight", &s3, &i3);
      t_layer = timer_read64();
      c3_nchw_f32_w8a32(l1, n, 32, 160, 160,
          w1, s1, i1, 16, W("model.2.cv1.conv.bias"),
          w2, s2, i2, 16, W("model.2.cv2.conv.bias"),
          w3, s3, i3, 32, W("model.2.cv3.conv.bias"),
          1, l2_cv1w, l2_cv1_scale, l2_cv1_is_int8, l2_cv1b, l2_cv2w, l2_cv2_scale, l2_cv2_is_int8, l2_cv2b, 1, l2);
      layer_cycles[2] = timer_delta64(t_layer, timer_read64());
    }
    LAYER_LOG_REF(2, layer_cycles[2], &l2[0]);
    yolo_timing_print_layer_ops(2);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l2, 16);
#endif
    feature_pool_free(l1);

    yolo_timing_set_layer(3);
    // Layer 3: Conv 3x3 s2
    POOL_ALLOC(l3, sz_l3);
    t_layer = timer_read64();
    { float _sw; int _iw; void* _pw = W_CONV("model.3.conv.weight", &_sw, &_iw);
      conv_block_nchw_f32_w8a32(l2, n, 32, 160, 160, _pw, _sw, _iw, 64, 3, 3, 2, 2, 1, 1,
          W("model.3.conv.bias"), l3, 80, 80); }
    layer_cycles[3] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(3, layer_cycles[3], &l3[0]);
    yolo_timing_print_layer_ops(3);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l3, 16);
#endif
    feature_pool_free(l2);

    yolo_timing_set_layer(4);
    // Layer 4: C3 (n=2)
    POOL_ALLOC(l4, sz_l4);
    float l4_cv1_scale[2]; int l4_cv1_is_int8[2]; const void* l4_cv1w[2]; l4_cv1w[0] = W_CONV("model.4.m.0.cv1.conv.weight", &l4_cv1_scale[0], &l4_cv1_is_int8[0]); l4_cv1w[1] = W_CONV("model.4.m.1.cv1.conv.weight", &l4_cv1_scale[1], &l4_cv1_is_int8[1]);
    float l4_cv2_scale[2]; int l4_cv2_is_int8[2]; const void* l4_cv2w[2]; l4_cv2w[0] = W_CONV("model.4.m.0.cv2.conv.weight", &l4_cv2_scale[0], &l4_cv2_is_int8[0]); l4_cv2w[1] = W_CONV("model.4.m.1.cv2.conv.weight", &l4_cv2_scale[1], &l4_cv2_is_int8[1]);
    const float* l4_cv1b[] = {W("model.4.m.0.cv1.conv.bias"), W("model.4.m.1.cv1.conv.bias")};
    const float* l4_cv2b[] = {W("model.4.m.0.cv2.conv.bias"), W("model.4.m.1.cv2.conv.bias")};
    { float s1, s2, s3; int i1, i2, i3; void* w1 = W_CONV("model.4.cv1.conv.weight", &s1, &i1); void* w2 = W_CONV("model.4.cv2.conv.weight", &s2, &i2); void* w3 = W_CONV("model.4.cv3.conv.weight", &s3, &i3);
      t_layer = timer_read64();
      c3_nchw_f32_w8a32(l3, n, 64, 80, 80, w1, s1, i1, 32, W("model.4.cv1.conv.bias"), w2, s2, i2, 32, W("model.4.cv2.conv.bias"), w3, s3, i3, 64, W("model.4.cv3.conv.bias"),
          2, l4_cv1w, l4_cv1_scale, l4_cv1_is_int8, l4_cv1b, l4_cv2w, l4_cv2_scale, l4_cv2_is_int8, l4_cv2b, 1, l4);
      layer_cycles[4] = timer_delta64(t_layer, timer_read64());
    }
    LAYER_LOG_REF(4, layer_cycles[4], &l4[0]);
    yolo_timing_print_layer_ops(4);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l4, 16);
#endif
    feature_pool_free(l3);

    yolo_timing_set_layer(5);
    // Layer 5: Conv 3x3 s2
    POOL_ALLOC(l5, sz_l5);
    t_layer = timer_read64();
    { float _sw; int _iw; void* _pw = W_CONV("model.5.conv.weight", &_sw, &_iw);
      conv_block_nchw_f32_w8a32(l4, n, 64, 80, 80, _pw, _sw, _iw, 128, 3, 3, 2, 2, 1, 1,
          W("model.5.conv.bias"), l5, 40, 40); }
    layer_cycles[5] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(5, layer_cycles[5], &l5[0]);
    yolo_timing_print_layer_ops(5);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l5, 16);
#endif

    yolo_timing_set_layer(6);
    // Layer 6: C3 (n=3)
    POOL_ALLOC(l6, sz_l6);
    float l6_cv1_scale[3]; int l6_cv1_is_int8[3]; const void* l6_cv1w[3]; l6_cv1w[0] = W_CONV("model.6.m.0.cv1.conv.weight", &l6_cv1_scale[0], &l6_cv1_is_int8[0]); l6_cv1w[1] = W_CONV("model.6.m.1.cv1.conv.weight", &l6_cv1_scale[1], &l6_cv1_is_int8[1]); l6_cv1w[2] = W_CONV("model.6.m.2.cv1.conv.weight", &l6_cv1_scale[2], &l6_cv1_is_int8[2]);
    float l6_cv2_scale[3]; int l6_cv2_is_int8[3]; const void* l6_cv2w[3]; l6_cv2w[0] = W_CONV("model.6.m.0.cv2.conv.weight", &l6_cv2_scale[0], &l6_cv2_is_int8[0]); l6_cv2w[1] = W_CONV("model.6.m.1.cv2.conv.weight", &l6_cv2_scale[1], &l6_cv2_is_int8[1]); l6_cv2w[2] = W_CONV("model.6.m.2.cv2.conv.weight", &l6_cv2_scale[2], &l6_cv2_is_int8[2]);
    const float* l6_cv1b[] = {W("model.6.m.0.cv1.conv.bias"), W("model.6.m.1.cv1.conv.bias"), W("model.6.m.2.cv1.conv.bias")};
    const float* l6_cv2b[] = {W("model.6.m.0.cv2.conv.bias"), W("model.6.m.1.cv2.conv.bias"), W("model.6.m.2.cv2.conv.bias")};
    { float s1, s2, s3; int i1, i2, i3; void* w1 = W_CONV("model.6.cv1.conv.weight", &s1, &i1); void* w2 = W_CONV("model.6.cv2.conv.weight", &s2, &i2); void* w3 = W_CONV("model.6.cv3.conv.weight", &s3, &i3);
      t_layer = timer_read64();
      c3_nchw_f32_w8a32(l5, n, 128, 40, 40, w1, s1, i1, 64, W("model.6.cv1.conv.bias"), w2, s2, i2, 64, W("model.6.cv2.conv.bias"), w3, s3, i3, 128, W("model.6.cv3.conv.bias"),
          3, l6_cv1w, l6_cv1_scale, l6_cv1_is_int8, l6_cv1b, l6_cv2w, l6_cv2_scale, l6_cv2_is_int8, l6_cv2b, 1, l6);
      layer_cycles[6] = timer_delta64(t_layer, timer_read64());
    }
    LAYER_LOG_REF(6, layer_cycles[6], &l6[0]);
    yolo_timing_print_layer_ops(6);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l6, 16);
#endif
    feature_pool_free(l5);

    yolo_timing_set_layer(7);
    // Layer 7: Conv 3x3 s2
    POOL_ALLOC(l7, sz_l7);
    t_layer = timer_read64();
    { float _sw; int _iw; void* _pw = W_CONV("model.7.conv.weight", &_sw, &_iw);
      conv_block_nchw_f32_w8a32(l6, n, 128, 40, 40, _pw, _sw, _iw, 256, 3, 3, 2, 2, 1, 1,
          W("model.7.conv.bias"), l7, 20, 20); }
    layer_cycles[7] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(7, layer_cycles[7], &l7[0]);
    yolo_timing_print_layer_ops(7);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l7, 16);
#endif

    yolo_timing_set_layer(8);
    // Layer 8: C3 (n=1)
    POOL_ALLOC(l8, sz_l8);
    float l8_cv1_scale[1]; int l8_cv1_is_int8[1]; const void* l8_cv1w[1]; l8_cv1w[0] = W_CONV("model.8.m.0.cv1.conv.weight", &l8_cv1_scale[0], &l8_cv1_is_int8[0]);
    float l8_cv2_scale[1]; int l8_cv2_is_int8[1]; const void* l8_cv2w[1]; l8_cv2w[0] = W_CONV("model.8.m.0.cv2.conv.weight", &l8_cv2_scale[0], &l8_cv2_is_int8[0]);
    const float* l8_cv1b[] = {W("model.8.m.0.cv1.conv.bias")};
    const float* l8_cv2b[] = {W("model.8.m.0.cv2.conv.bias")};
    { float s1, s2, s3; int i1, i2, i3; void* w1 = W_CONV("model.8.cv1.conv.weight", &s1, &i1); void* w2 = W_CONV("model.8.cv2.conv.weight", &s2, &i2); void* w3 = W_CONV("model.8.cv3.conv.weight", &s3, &i3);
      t_layer = timer_read64();
      c3_nchw_f32_w8a32(l7, n, 256, 20, 20, w1, s1, i1, 128, W("model.8.cv1.conv.bias"), w2, s2, i2, 128, W("model.8.cv2.conv.bias"), w3, s3, i3, 256, W("model.8.cv3.conv.bias"),
          1, l8_cv1w, l8_cv1_scale, l8_cv1_is_int8, l8_cv1b, l8_cv2w, l8_cv2_scale, l8_cv2_is_int8, l8_cv2b, 1, l8);
      layer_cycles[8] = timer_delta64(t_layer, timer_read64());
    }
    LAYER_LOG_REF(8, layer_cycles[8], &l8[0]);
    yolo_timing_print_layer_ops(8);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l8, 16);
#endif
    feature_pool_free(l7);

    yolo_timing_set_layer(9);
    // Layer 9: SPPF
    POOL_ALLOC(l9, sz_l9);
    t_layer = timer_read64();
    { float s1, s2; int i1, i2; void* w1 = W_CONV("model.9.cv1.conv.weight", &s1, &i1); void* w2 = W_CONV("model.9.cv2.conv.weight", &s2, &i2);
      sppf_nchw_f32_w8a32(l8, n, 256, 20, 20,
          w1, s1, i1, 128, W("model.9.cv1.conv.bias"),
          w2, s2, i2, 256, W("model.9.cv2.conv.bias"),
          5, l9); }
    layer_cycles[9] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(9, layer_cycles[9], &l9[0]);
    yolo_timing_print_layer_ops(9);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l9, 16);
#endif
    feature_pool_free(l8);
    cycles_backbone = timer_delta64(t_stage_start, timer_read64());

    // ===== Neck =====
    YOLO_LOG("\nNeck: ");
    t_stage_start = timer_read64();
    yolo_timing_set_layer(10);
    // Layer 10: Conv 1x1
    POOL_ALLOC(l10, sz_l10);
    t_layer = timer_read64();
    { float _sw; int _iw; void* _pw = W_CONV("model.10.conv.weight", &_sw, &_iw);
      conv_block_nchw_f32_w8a32(l9, n, 256, 20, 20, _pw, _sw, _iw, 128, 1, 1, 1, 1, 0, 0,
          W("model.10.conv.bias"), l10, 20, 20); }
    layer_cycles[10] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(10, layer_cycles[10], &l10[0]);
    yolo_timing_print_layer_ops(10);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l10, 16);
#endif
    feature_pool_free(l9);

    yolo_timing_set_layer(11);
    // Layer 11: Upsample
    POOL_ALLOC(l11, sz_l11);
    t_layer = timer_read64();
    upsample_nearest2x_nchw_f32_w8a32(l10, n, 128, 20, 20, l11);
    layer_cycles[11] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(11, layer_cycles[11], &l11[0]);
    yolo_timing_print_layer_ops(11);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l11, 16);
#endif

    yolo_timing_set_layer(12);
    // Layer 12: Concat (l11 + l6)
    POOL_ALLOC(l12, sz_l12);
    t_layer = timer_read64();
    yolo_timing_begin("concat");
    concat_nchw_f32_w8a32(l11, 128, l6, 128, n, 40, 40, l12);
    yolo_timing_end();
    layer_cycles[12] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(12, layer_cycles[12], &l12[0]);
    yolo_timing_print_layer_ops(12);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l12, 16);
#endif
    feature_pool_free(l11);
    feature_pool_free(l6);

    yolo_timing_set_layer(13);
    // Layer 13: C3 (n=1)
    POOL_ALLOC(l13, sz_l13);
    float l13_cv1_scale[1]; int l13_cv1_is_int8[1]; const void* l13_cv1w[1]; l13_cv1w[0] = W_CONV("model.13.m.0.cv1.conv.weight", &l13_cv1_scale[0], &l13_cv1_is_int8[0]);
    float l13_cv2_scale[1]; int l13_cv2_is_int8[1]; const void* l13_cv2w[1]; l13_cv2w[0] = W_CONV("model.13.m.0.cv2.conv.weight", &l13_cv2_scale[0], &l13_cv2_is_int8[0]);
    const float* l13_cv1b[] = {W("model.13.m.0.cv1.conv.bias")};
    const float* l13_cv2b[] = {W("model.13.m.0.cv2.conv.bias")};
    { float s1, s2, s3; int i1, i2, i3; void* w1 = W_CONV("model.13.cv1.conv.weight", &s1, &i1); void* w2 = W_CONV("model.13.cv2.conv.weight", &s2, &i2); void* w3 = W_CONV("model.13.cv3.conv.weight", &s3, &i3);
      t_layer = timer_read64();
      c3_nchw_f32_w8a32(l12, n, 256, 40, 40, w1, s1, i1, 64, W("model.13.cv1.conv.bias"), w2, s2, i2, 64, W("model.13.cv2.conv.bias"), w3, s3, i3, 128, W("model.13.cv3.conv.bias"),
          1, l13_cv1w, l13_cv1_scale, l13_cv1_is_int8, l13_cv1b, l13_cv2w, l13_cv2_scale, l13_cv2_is_int8, l13_cv2b, 0, l13);
      layer_cycles[13] = timer_delta64(t_layer, timer_read64());
    }
    LAYER_LOG_REF(13, layer_cycles[13], &l13[0]);
    yolo_timing_print_layer_ops(13);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l13, 16);
#endif
    feature_pool_free(l12);

    yolo_timing_set_layer(14);
    // Layer 14: Conv 1x1
    POOL_ALLOC(l14, sz_l14);
    t_layer = timer_read64();
    { float _sw; int _iw; void* _pw = W_CONV("model.14.conv.weight", &_sw, &_iw);
      conv_block_nchw_f32_w8a32(l13, n, 128, 40, 40, _pw, _sw, _iw, 64, 1, 1, 1, 1, 0, 0,
          W("model.14.conv.bias"), l14, 40, 40); }
    layer_cycles[14] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(14, layer_cycles[14], &l14[0]);
    yolo_timing_print_layer_ops(14);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l14, 16);
#endif
    feature_pool_free(l13);

    yolo_timing_set_layer(15);
    // Layer 15: Upsample
    POOL_ALLOC(l15, sz_l15);
    t_layer = timer_read64();
    upsample_nearest2x_nchw_f32_w8a32(l14, n, 64, 40, 40, l15);
    layer_cycles[15] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(15, layer_cycles[15], &l15[0]);
    yolo_timing_print_layer_ops(15);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l15, 16);
#endif

    yolo_timing_set_layer(16);
    // Layer 16: Concat (l15 + l4)
    POOL_ALLOC(l16, sz_l16);
    t_layer = timer_read64();
    yolo_timing_begin("concat");
    concat_nchw_f32_w8a32(l15, 64, l4, 64, n, 80, 80, l16);
    yolo_timing_end();
    layer_cycles[16] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(16, layer_cycles[16], &l16[0]);
    yolo_timing_print_layer_ops(16);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l16, 16);
#endif
    feature_pool_free(l15);
    feature_pool_free(l4);

    yolo_timing_set_layer(17);
    // Layer 17: C3 (n=1) -> P3
    POOL_ALLOC(l17, sz_l17);
    float l17_cv1_scale[1]; int l17_cv1_is_int8[1]; const void* l17_cv1w[1]; l17_cv1w[0] = W_CONV("model.17.m.0.cv1.conv.weight", &l17_cv1_scale[0], &l17_cv1_is_int8[0]);
    float l17_cv2_scale[1]; int l17_cv2_is_int8[1]; const void* l17_cv2w[1]; l17_cv2w[0] = W_CONV("model.17.m.0.cv2.conv.weight", &l17_cv2_scale[0], &l17_cv2_is_int8[0]);
    const float* l17_cv1b[] = {W("model.17.m.0.cv1.conv.bias")};
    const float* l17_cv2b[] = {W("model.17.m.0.cv2.conv.bias")};
    { float s1, s2, s3; int i1, i2, i3; void* w1 = W_CONV("model.17.cv1.conv.weight", &s1, &i1); void* w2 = W_CONV("model.17.cv2.conv.weight", &s2, &i2); void* w3 = W_CONV("model.17.cv3.conv.weight", &s3, &i3);
      t_layer = timer_read64();
      c3_nchw_f32_w8a32(l16, n, 128, 80, 80, w1, s1, i1, 32, W("model.17.cv1.conv.bias"), w2, s2, i2, 32, W("model.17.cv2.conv.bias"), w3, s3, i3, 64, W("model.17.cv3.conv.bias"),
          1, l17_cv1w, l17_cv1_scale, l17_cv1_is_int8, l17_cv1b, l17_cv2w, l17_cv2_scale, l17_cv2_is_int8, l17_cv2b, 0, l17);
      layer_cycles[17] = timer_delta64(t_layer, timer_read64());
    }
    LAYER_LOG_REF(17, layer_cycles[17], &l17[0]);
    yolo_timing_print_layer_ops(17);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l17, 16);
#endif
    feature_pool_free(l16);

    yolo_timing_set_layer(18);
    // Layer 18: Conv 3x3 s2
    POOL_ALLOC(l18, sz_l18);
    t_layer = timer_read64();
    { float _sw; int _iw; void* _pw = W_CONV("model.18.conv.weight", &_sw, &_iw);
      conv_block_nchw_f32_w8a32(l17, n, 64, 80, 80, _pw, _sw, _iw, 64, 3, 3, 2, 2, 1, 1,
          W("model.18.conv.bias"), l18, 40, 40); }
    layer_cycles[18] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(18, layer_cycles[18], &l18[0]);
    yolo_timing_print_layer_ops(18);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l18, 16);
#endif

    yolo_timing_set_layer(19);
    // Layer 19: Concat (l18 + l14)
    POOL_ALLOC(l19, sz_l19);
    t_layer = timer_read64();
    yolo_timing_begin("concat");
    concat_nchw_f32_w8a32(l18, 64, l14, 64, n, 40, 40, l19);
    yolo_timing_end();
    layer_cycles[19] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(19, layer_cycles[19], &l19[0]);
    yolo_timing_print_layer_ops(19);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l19, 16);
#endif
    feature_pool_free(l18);
    feature_pool_free(l14);

    yolo_timing_set_layer(20);
    // Layer 20: C3 (n=1) -> P4
    POOL_ALLOC(l20, sz_l20);
    float l20_cv1_scale[1]; int l20_cv1_is_int8[1]; const void* l20_cv1w[1]; l20_cv1w[0] = W_CONV("model.20.m.0.cv1.conv.weight", &l20_cv1_scale[0], &l20_cv1_is_int8[0]);
    float l20_cv2_scale[1]; int l20_cv2_is_int8[1]; const void* l20_cv2w[1]; l20_cv2w[0] = W_CONV("model.20.m.0.cv2.conv.weight", &l20_cv2_scale[0], &l20_cv2_is_int8[0]);
    const float* l20_cv1b[] = {W("model.20.m.0.cv1.conv.bias")};
    const float* l20_cv2b[] = {W("model.20.m.0.cv2.conv.bias")};
    { float s1, s2, s3; int i1, i2, i3; void* w1 = W_CONV("model.20.cv1.conv.weight", &s1, &i1); void* w2 = W_CONV("model.20.cv2.conv.weight", &s2, &i2); void* w3 = W_CONV("model.20.cv3.conv.weight", &s3, &i3);
      t_layer = timer_read64();
      c3_nchw_f32_w8a32(l19, n, 128, 40, 40, w1, s1, i1, 64, W("model.20.cv1.conv.bias"), w2, s2, i2, 64, W("model.20.cv2.conv.bias"), w3, s3, i3, 128, W("model.20.cv3.conv.bias"),
          1, l20_cv1w, l20_cv1_scale, l20_cv1_is_int8, l20_cv1b, l20_cv2w, l20_cv2_scale, l20_cv2_is_int8, l20_cv2b, 0, l20);
      layer_cycles[20] = timer_delta64(t_layer, timer_read64());
    }
    LAYER_LOG_REF(20, layer_cycles[20], &l20[0]);
    yolo_timing_print_layer_ops(20);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l20, 16);
#endif
    feature_pool_free(l19);

    yolo_timing_set_layer(21);
    // Layer 21: Conv 3x3 s2
    POOL_ALLOC(l21, sz_l21);
    t_layer = timer_read64();
    { float _sw; int _iw; void* _pw = W_CONV("model.21.conv.weight", &_sw, &_iw);
      conv_block_nchw_f32_w8a32(l20, n, 128, 40, 40, _pw, _sw, _iw, 128, 3, 3, 2, 2, 1, 1,
          W("model.21.conv.bias"), l21, 20, 20); }
    layer_cycles[21] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(21, layer_cycles[21], &l21[0]);
    yolo_timing_print_layer_ops(21);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l21, 16);
#endif

    yolo_timing_set_layer(22);
    // Layer 22: Concat (l21 + l10)
    POOL_ALLOC(l22, sz_l22);
    t_layer = timer_read64();
    yolo_timing_begin("concat");
    concat_nchw_f32_w8a32(l21, 128, l10, 128, n, 20, 20, l22);
    yolo_timing_end();
    layer_cycles[22] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG_REF(22, layer_cycles[22], &l22[0]);
    yolo_timing_print_layer_ops(22);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l22, 16);
#endif
    feature_pool_free(l21);
    feature_pool_free(l10);

    yolo_timing_set_layer(23);
    // Layer 23: C3 (n=1) -> P5
    POOL_ALLOC(l23, sz_l23);
    float l23_cv1_scale[1]; int l23_cv1_is_int8[1]; const void* l23_cv1w[1]; l23_cv1w[0] = W_CONV("model.23.m.0.cv1.conv.weight", &l23_cv1_scale[0], &l23_cv1_is_int8[0]);
    float l23_cv2_scale[1]; int l23_cv2_is_int8[1]; const void* l23_cv2w[1]; l23_cv2w[0] = W_CONV("model.23.m.0.cv2.conv.weight", &l23_cv2_scale[0], &l23_cv2_is_int8[0]);
    const float* l23_cv1b[] = {W("model.23.m.0.cv1.conv.bias")};
    const float* l23_cv2b[] = {W("model.23.m.0.cv2.conv.bias")};
    { float s1, s2, s3; int i1, i2, i3; void* w1 = W_CONV("model.23.cv1.conv.weight", &s1, &i1); void* w2 = W_CONV("model.23.cv2.conv.weight", &s2, &i2); void* w3 = W_CONV("model.23.cv3.conv.weight", &s3, &i3);
      t_layer = timer_read64();
      c3_nchw_f32_w8a32(l22, n, 256, 20, 20, w1, s1, i1, 128, W("model.23.cv1.conv.bias"), w2, s2, i2, 128, W("model.23.cv2.conv.bias"), w3, s3, i3, 256, W("model.23.cv3.conv.bias"),
          1, l23_cv1w, l23_cv1_scale, l23_cv1_is_int8, l23_cv1b, l23_cv2w, l23_cv2_scale, l23_cv2_is_int8, l23_cv2b, 0, l23);
      layer_cycles[23] = timer_delta64(t_layer, timer_read64());
    }
    LAYER_LOG_REF(23, layer_cycles[23], &l23[0]);
    yolo_timing_print_layer_ops(23);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l23, 16);
#endif
    feature_pool_free(l22);
    cycles_neck = timer_delta64(t_stage_start, timer_read64());

    // ===== Detect Head =====
    YOLO_LOG("\nHead: ");
    yolo_timing_set_layer(24);
    t_stage_start = timer_read64();
#ifdef BARE_METAL
    (void)sz_p3;
    (void)sz_p4;
    (void)sz_p5;
    p3 = (float*)DETECT_HEAD_BASE;
    p4 = p3 + (DETECT_C_OUT * 80 * 80);
    p5 = p4 + (DETECT_C_OUT * 40 * 40);
#else
    POOL_ALLOC(p3, sz_p3);
    POOL_ALLOC(p4, sz_p4);
    POOL_ALLOC(p5, sz_p5);
#endif
#undef POOL_ALLOC
    { float s0, s1, s2; int i0, i1, i2;
      void* m0 = W_CONV("model.24.m.0.weight", &s0, &i0); void* m1 = W_CONV("model.24.m.1.weight", &s1, &i1); void* m2 = W_CONV("model.24.m.2.weight", &s2, &i2);
      detect_nchw_f32_w8a32(
          l17, 64, 80, 80, l20, 128, 40, 40, l23, 256, 20, 20,
          m0, s0, i0, W("model.24.m.0.bias"),
          m1, s1, i1, W("model.24.m.1.bias"),
          m2, s2, i2, W("model.24.m.2.bias"),
          p3, p4, p5);
    }
    YOLO_LOG("Detect\n");
    cycles_head = timer_delta64(t_stage_start, timer_read64());
#ifdef BARE_METAL
    YOLO_LOG("  det %llu ms\n", LAYER_MS_INT(cycles_head));
#else
    YOLO_LOG("  det %.2f ms\n", LAYER_MS(cycles_head));
#endif
    yolo_timing_print_layer_ops(24);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)DETECT_HEAD_BASE, (unsigned int)DETECT_HEAD_SIZE);
    __sync_synchronize();
#endif
    feature_pool_free(l17);
    feature_pool_free(l20);
    feature_pool_free(l23);
#ifndef BARE_METAL
    feature_pool_free(p3);
    feature_pool_free(p4);
    feature_pool_free(p5);
#endif
#endif /* !USE_W8A16 */

    // ===== Decode =====
    yolo_timing_set_layer(25);
    t_stage_start = timer_read64();
    detection_t* dets = malloc(MAX_DETECTIONS * sizeof(detection_t));
    int32_t num_dets = decode_nchw_f32(
        p3, 80, 80, p4, 40, 40, p5, 20, 20,
        NUM_CLASSES, CONF_THRESHOLD, INPUT_SIZE, STRIDES, ANCHORS,
        dets, MAX_DETECTIONS);

    YOLO_LOG("Decoded: %d detections\n", num_dets);
    cycles_decode = timer_delta64(t_stage_start, timer_read64());
#ifdef BARE_METAL
    YOLO_LOG("  dec %llu ms\n", LAYER_MS_INT(cycles_decode));
#else
    YOLO_LOG("  dec %.2f ms\n", LAYER_MS(cycles_decode));
#endif
    yolo_timing_print_layer_ops(25);
    if (YOLO_DEBUG && p3) {
        union { float f; uint32_t u; } u0 = { .f = p3[0] }, u1 = { .f = p3[1] }, u4 = { .f = p3[4 * 80 * 80] };
        YOLO_LOG("DEBUG p3[0]=0x%08X p3[1]=0x%08X p3[obj0]=0x%08X\n", (unsigned)u0.u, (unsigned)u1.u, (unsigned)u4.u);
    }

    // Sort by confidence
    for (int i = 0; i < num_dets - 1; i++) {
        for (int j = i + 1; j < num_dets; j++) {
            if (dets[i].conf < dets[j].conf) {
                detection_t t = dets[i]; dets[i] = dets[j]; dets[j] = t;
            }
        }
    }

    // NMS
    yolo_timing_set_layer(26);
    t_stage_start = timer_read64();
    detection_t* nms_dets = NULL;
    int32_t num_nms = 0;
    nms(dets, num_dets, &nms_dets, &num_nms, IOU_THRESHOLD, MAX_DETECTIONS);
    cycles_nms = timer_delta64(t_stage_start, timer_read64());
#ifdef BARE_METAL
    YOLO_LOG("  nms %llu ms\n", LAYER_MS_INT(cycles_nms));
#else
    YOLO_LOG("  nms %.2f ms\n", LAYER_MS(cycles_nms));
#endif
    yolo_timing_print_layer_ops(26);
    {
        uint64_t total = timer_delta64(t_total_start, timer_read64());
#ifdef BARE_METAL
        {
            YOLO_LOG("[mcycle] backbone=%llu neck=%llu head=%llu decode=%llu nms=%llu total=%llu\n",
                     (unsigned long long)cycles_backbone, (unsigned long long)cycles_neck,
                     (unsigned long long)cycles_head, (unsigned long long)cycles_decode,
                     (unsigned long long)cycles_nms, (unsigned long long)total);
            YOLO_LOG("[time @ %dMHz] backbone=%llu neck=%llu head=%llu decode=%llu nms=%llu total=%llu ms\n",
                     (int)CPU_MHZ, LAYER_MS_INT(cycles_backbone), LAYER_MS_INT(cycles_neck),
                     LAYER_MS_INT(cycles_head), LAYER_MS_INT(cycles_decode), LAYER_MS_INT(cycles_nms),
                     LAYER_MS_INT(total));
        }
#else
        YOLO_LOG("[time] backbone=%.2f ms neck=%.2f ms head=%.2f ms decode=%.2f ms nms=%.2f ms total=%.2f ms\n",
                 cycles_backbone / 1000.0, cycles_neck / 1000.0, cycles_head / 1000.0,
                 cycles_decode / 1000.0, cycles_nms / 1000.0, total / 1000.0);
#endif
    }
    YOLO_LOG("After NMS: %d detections\n", num_nms);

    {
        uint8_t count = (uint8_t)(num_nms > 255 ? 255 : num_nms);
#ifdef BARE_METAL
        uint8_t* out = (uint8_t*)DETECTIONS_OUT_BASE;
        *out++ = count;
        for (int i = 0; i < count; i++) {
            hw_detection_t hw;
            hw.x = (uint16_t)(nms_dets[i].x * INPUT_SIZE);
            hw.y = (uint16_t)(nms_dets[i].y * INPUT_SIZE);
            hw.w = (uint16_t)(nms_dets[i].w * INPUT_SIZE);
            hw.h = (uint16_t)(nms_dets[i].h * INPUT_SIZE);
            hw.class_id = (uint8_t)nms_dets[i].cls_id;
            hw.confidence = (uint8_t)(nms_dets[i].conf * 255);
            hw.reserved[0] = 0;
            hw.reserved[1] = 0;
            memcpy(out, &hw, sizeof(hw_detection_t));
            out += sizeof(hw_detection_t);
        }
        YOLO_LOG("Sending %d detections to UART...\n", (int)count);
        yolo_uart_send_detections((const void*)((uint8_t*)DETECTIONS_OUT_BASE + 1), count);
        YOLO_LOG("Done. Results at DDR 0x%08X\n", (unsigned int)DETECTIONS_OUT_BASE);
        Xil_DCacheEnable();
#else
        FILE* f = fopen("data/output/detections.bin", "wb");
        if (f) {
            fwrite(&count, sizeof(uint8_t), 1, f);
            for (int i = 0; i < count; i++) {
                hw_detection_t hw;
                hw.x = (uint16_t)(nms_dets[i].x * INPUT_SIZE);
                hw.y = (uint16_t)(nms_dets[i].y * INPUT_SIZE);
                hw.w = (uint16_t)(nms_dets[i].w * INPUT_SIZE);
                hw.h = (uint16_t)(nms_dets[i].h * INPUT_SIZE);
                hw.class_id = (uint8_t)nms_dets[i].cls_id;
                hw.confidence = (uint8_t)(nms_dets[i].conf * 255);
                hw.reserved[0] = 0;
                hw.reserved[1] = 0;
                fwrite(&hw, sizeof(hw_detection_t), 1, f);
            }
            fclose(f);
            printf("Saved to data/output/detections.bin (%d bytes)\n",
                   1 + count * (int)sizeof(hw_detection_t));
        }
#endif
        YOLO_LOG("Summary: %d | ", (int)count);
        for (int i = 0; i < (int)count; i++) {
            int cls = nms_dets[i].cls_id;
            const char* name = (cls >= 0 && cls < NUM_CLASSES) ? COCO_NAMES[cls] : "?";
            int pct = (int)(nms_dets[i].conf * 100);
            int px = (int)(nms_dets[i].x * (float)INPUT_SIZE);
            int py = (int)(nms_dets[i].y * (float)INPUT_SIZE);
            YOLO_LOG("%s %d%% (%d,%d)%s", name, pct, px, py, (i < (int)count - 1) ? " | " : "");
        }
        YOLO_LOG("\n");
    }
    free(dets);
    if (nms_dets) free(nms_dets);
#ifdef USE_W8A16
    free(p3);
    free(p4);
    free(p5);
#endif
    feature_pool_reset();
    weights_free(&weights);
    image_free(&img);

    return 0;
}
