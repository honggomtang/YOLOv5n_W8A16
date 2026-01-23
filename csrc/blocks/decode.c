#include "decode.h"
#include <math.h>
#include <stdlib.h>

// Sigmoid 함수
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Softmax 함수 (16개 채널에 대해)
static void softmax_16(const float* x, float* out) {
    float max_val = x[0];
    for (int i = 1; i < 16; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        out[i] = expf(x[i] - max_val);  // 수치 안정성을 위해 max_val 빼기
        sum += out[i];
    }
    
    for (int i = 0; i < 16; i++) {
        out[i] /= sum;
    }
}

// DFL 연산: 16개 채널에 Softmax를 취한 뒤, 0~15까지의 가중치를 곱해 합산
static float dfl_decode(const float* channels) {
    float softmax_out[16];
    softmax_16(channels, softmax_out);
    
    float dist = 0.0f;
    for (int i = 0; i < 16; i++) {
        dist += softmax_out[i] * (float)i;
    }
    return dist;
}

// Decode: cv2 (bbox)와 cv3 (class) 출력을 받아서 detection 결과로 변환
int32_t decode_detections_nchw_f32(
    // P3 cv2, cv3 출력
    const float* p3_cv2, int32_t p3_h, int32_t p3_w,
    const float* p3_cv3, int32_t p3_cv3_c,
    // P4 cv2, cv3 출력
    const float* p4_cv2, int32_t p4_h, int32_t p4_w,
    const float* p4_cv3, int32_t p4_cv3_c,
    // P5 cv2, cv3 출력
    const float* p5_cv2, int32_t p5_h, int32_t p5_w,
    const float* p5_cv3, int32_t p5_cv3_c,
    // 파라미터
    int32_t num_classes,
    float conf_threshold,
    int32_t input_size,
    const float strides[3],
    // 출력
    detection_t* detections,
    int32_t max_detections)
{
    int32_t count = 0;
    
    // 각 scale (P3, P4, P5)에 대해 decode
    for (int scale = 0; scale < 3; scale++) {
        const float* cv2 = NULL;
        const float* cv3 = NULL;
        int32_t grid_h = 0, grid_w = 0;
        float stride = 0.0f;
        
        switch (scale) {
            case 0:  // P3
                cv2 = p3_cv2;
                cv3 = p3_cv3;
                grid_h = p3_h;
                grid_w = p3_w;
                stride = strides[0];
                break;
            case 1:  // P4
                cv2 = p4_cv2;
                cv3 = p4_cv3;
                grid_h = p4_h;
                grid_w = p4_w;
                stride = strides[1];
                break;
            case 2:  // P5
                cv2 = p5_cv2;
                cv3 = p5_cv3;
                grid_h = p5_h;
                grid_w = p5_w;
                stride = strides[2];
                break;
        }
        
        if (!cv2 || !cv3) continue;
        
        // 각 grid cell에 대해 (anchor-free 구조)
        for (int32_t y = 0; y < grid_h; y++) {
            for (int32_t x = 0; x < grid_w; x++) {
                const int32_t spatial_idx = y * grid_w + x;
                const int32_t grid_size = grid_h * grid_w;
                
                // cv3에서 class 추출
                // cv3: (1, 80, H, W) - 각 grid cell당 80개 class
                const int32_t cv3_base = spatial_idx;
                float max_cls_conf = 0.0f;
                int32_t max_cls_id = 0;
                for (int c = 0; c < num_classes; c++) {
                    const float cls_logit = cv3[cv3_base + c * grid_size];
                    const float cls_conf = sigmoid(cls_logit);
                    if (cls_conf > max_cls_conf) {
                        max_cls_conf = cls_conf;
                        max_cls_id = c;
                    }
                }
                
                // cv2에서 bbox 추출 (DFL 구조)
                // cv2: (1, 64, H, W)
                // 64채널 = 4개 좌표(Left, Top, Right, Bottom) × 16채널(reg_max=16)
                // Channel 0~15: Left (L)
                // Channel 16~31: Top (T)
                // Channel 32~47: Right (R)
                // Channel 48~63: Bottom (B)
                
                const int32_t cv2_base = spatial_idx;
                const int32_t cv2_channel_stride = grid_size;
                
                // 각 좌표의 16개 채널 추출
                float left_channels[16];
                float top_channels[16];
                float right_channels[16];
                float bottom_channels[16];
                
                for (int i = 0; i < 16; i++) {
                    left_channels[i] = cv2[cv2_base + (0 + i) * cv2_channel_stride];
                    top_channels[i] = cv2[cv2_base + (16 + i) * cv2_channel_stride];
                    right_channels[i] = cv2[cv2_base + (32 + i) * cv2_channel_stride];
                    bottom_channels[i] = cv2[cv2_base + (48 + i) * cv2_channel_stride];
                }
                
                // DFL 연산: 각 좌표의 16개 채널에 Softmax를 취한 뒤, 0~15까지의 가중치를 곱해 합산
                const float dist_L = dfl_decode(left_channels);
                const float dist_T = dfl_decode(top_channels);
                const float dist_R = dfl_decode(right_channels);
                const float dist_B = dfl_decode(bottom_channels);
                
                // 최종 좌표 복원
                // x1 = (grid_x - dist_L) * stride
                // y1 = (grid_y - dist_T) * stride
                // x2 = (grid_x + dist_R) * stride
                // y2 = (grid_y + dist_B) * stride
                const float grid_x = (float)x;
                const float grid_y = (float)y;
                
                const float x1 = (grid_x - dist_L) * stride;
                const float y1 = (grid_y - dist_T) * stride;
                const float x2 = (grid_x + dist_R) * stride;
                const float y2 = (grid_y + dist_B) * stride;
                
                // Center + Width/Height 형태로 변환
                const float center_x = (x1 + x2) / 2.0f;
                const float center_y = (y1 + y2) / 2.0f;
                const float width = x2 - x1;
                const float height = y2 - y1;
                
                // Class Score 결합: cv3에서 나온 클래스 확률 사용
                // obj_conf는 cv2에 없으므로 class confidence만 사용
                const float conf = max_cls_conf;
                
                if (conf < conf_threshold) continue;
                if (count >= max_detections) goto done;
                
                // Normalize to 0-1 range
                detections[count].x = center_x / input_size;
                detections[count].y = center_y / input_size;
                detections[count].w = width / input_size;
                detections[count].h = height / input_size;
                detections[count].conf = conf;
                detections[count].cls_id = max_cls_id;
                
                count++;
            }
        }
    }
    
done:
    return count;
}
