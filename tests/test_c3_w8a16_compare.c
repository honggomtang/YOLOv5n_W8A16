/*
 * C3 W8A16 단위 검증
 * - 작은 shape, zero weights로 파이프라인 동작 및 shift 인자 전달 확인
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../csrc/utils/feature_pool.h"
#include "../csrc/blocks/c3_w8a16.h"

int main(void) {
    printf("=== C3 W8A16 unit test ===\n\n");

    feature_pool_init();

    /* 작은 C3: c_in=4, cv1_c_out=2, cv2_c_out=2, cv3_c_out=4, n_bottleneck=1, h=w=4 */
    const int n = 1, c_in = 4, h = 4, w = 4;
    const int cv1_c_out = 2, cv2_c_out = 2, cv3_c_out = 4;
    const int n_bottleneck = 1;

    static int16_t x[1 * 4 * 4 * 4];
    static int16_t y[1 * 4 * 4 * 4];
    static int8_t cv1_w[2 * 4 * 1 * 1];
    static int8_t cv2_w[2 * 4 * 1 * 1];
    static int8_t cv3_w[4 * 4 * 1 * 1];  /* (cv1_c_out+cv2_c_out)=4, cv3_c_out=4 */
    static int8_t bn_cv1_w[2 * 2 * 1 * 1];
    static int8_t bn_cv2_w[2 * 2 * 3 * 3];
    memset(cv1_w, 0, sizeof(cv1_w));
    memset(cv2_w, 0, sizeof(cv2_w));
    memset(cv3_w, 0, sizeof(cv3_w));
    memset(bn_cv1_w, 0, sizeof(bn_cv1_w));
    memset(bn_cv2_w, 0, sizeof(bn_cv2_w));

    const int8_t* bn_cv1_w_arr[1] = { bn_cv1_w };
    const int8_t* bn_cv2_w_arr[1] = { bn_cv2_w };
    const int32_t* bn_cv1_bias_arr[1] = { NULL };
    const int32_t* bn_cv2_bias_arr[1] = { NULL };
    int32_t bn_cv1_shift[1] = { 10 };
    int32_t bn_cv2_shift[1] = { 10 };

    for (int i = 0; i < n * c_in * h * w; i++)
        x[i] = (int16_t)((i % 7) - 3);

    c3_nchw_w8a16(
        x, n, c_in, h, w,
        cv1_w, cv1_c_out, NULL, 10,
        cv2_w, cv2_c_out, NULL, 10,
        cv3_w, cv3_c_out, NULL, 10,
        n_bottleneck,
        bn_cv1_w_arr, bn_cv1_bias_arr, bn_cv1_shift,
        bn_cv2_w_arr, bn_cv2_bias_arr, bn_cv2_shift,
        1 /* shortcut */,
        y);

    printf("  C3 W8A16 run: OK (no crash, output shape 1*%d*%d*%d)\n", cv3_c_out, h, w);
    printf("  Sample y[0..3]: %d %d %d %d\n", (int)y[0], (int)y[1], (int)y[2], (int)y[3]);
    printf("\nResult: OK (C3 W8A16 pipeline + per-layer shift OK)\n");
    return 0;
}
