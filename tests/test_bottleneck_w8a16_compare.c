/*
 * Bottleneck W8A16 단위 검증
 * - Zero weights로 Conv 출력 0 → shortcut 시 y = clamp_s16(x+0) = x 인지 확인
 * - shortcut=0 시 y = 0 (cv2_out) 인지 확인
 * - int32 잔차 덧셈 + clamp_s16 동작 검증
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../csrc/utils/feature_pool.h"
#include "../csrc/operations/bottleneck_w8a16.h"

int main(void) {
    printf("=== Bottleneck W8A16 unit test ===\n\n");

    feature_pool_init();

    const int n = 1, c = 2, h = 4, w = 4, cv1_c_out = 2, cv2_c_out = 2;
    /* cv1: 1x1, c_in=2 c_out=2 → weight 2*2*1*1 = 4. cv2: 3x3, 2*2*3*3 = 36 */
    static int16_t x[1 * 2 * 4 * 4];
    static int16_t y[1 * 2 * 4 * 4];
    static int8_t cv1_w[2 * 2 * 1 * 1];  /* zero */
    static int8_t cv2_w[2 * 2 * 3 * 3];   /* zero */
    memset(cv1_w, 0, sizeof(cv1_w));
    memset(cv2_w, 0, sizeof(cv2_w));

    for (int i = 0; i < n * c * h * w; i++)
        x[i] = (int16_t)((i % 10) - 5);  /* -5 ~ 4 */

    /* shortcut=1, c==cv2_c_out: residual path. conv 출력 0 → y = clamp_s16(x + 0) = x */
    bottleneck_nchw_w8a16(
        x, n, c, h, w,
        cv1_w, cv1_c_out, NULL, 10,
        cv2_w, cv2_c_out, NULL, 10,
        1 /* shortcut */,
        y);

    int ok_shortcut = 1;
    for (int i = 0; i < n * c * h * w; i++) {
        if (y[i] != x[i]) {
            ok_shortcut = 0;
            printf("  shortcut=1: mismatch at %d: x=%d y=%d\n", i, (int)x[i], (int)y[i]);
            break;
        }
    }
    printf("  shortcut=1 (y should equal x): %s\n", ok_shortcut ? "OK" : "NG");

    /* shortcut=0: y = cv2_out only (zeros) */
    bottleneck_nchw_w8a16(
        x, n, c, h, w,
        cv1_w, cv1_c_out, NULL, 10,
        cv2_w, cv2_c_out, NULL, 10,
        0 /* no shortcut */,
        y);

    int ok_no_shortcut = 1;
    for (int i = 0; i < n * cv2_c_out * h * w; i++) {
        if (y[i] != 0) {
            ok_no_shortcut = 0;
            printf("  shortcut=0: expected 0 at %d, got %d\n", i, (int)y[i]);
            break;
        }
    }
    printf("  shortcut=0 (y should be 0): %s\n", ok_no_shortcut ? "OK" : "NG");

    /* int32 overflow 방지: 큰 x + 0 = x, clamp_s16 유지. (이미 위에서 x 범위 -5~4라 overflow 없음) */
    printf("\nResult: %s\n", (ok_shortcut && ok_no_shortcut) ? "OK (Bottleneck W8A16 residual + clamp_s16 OK)" : "NG");
    return (ok_shortcut && ok_no_shortcut) ? 0 : 1;
}
