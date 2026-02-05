# W8A16 구현 정리

YOLOv5n 전체 추론을 **INT8 가중치 + INT16 활성화(Q6.10 고정소수점)** 로 수행하는 경로의 개념·데이터 형식·코드 구조를 정리한 문서이다. 시행착오 없이 **최종 구현 결과**만 기술한다.

---

## 1. 개념

- **W8A16**: 가중치 8비트(INT8), 활성화 16비트(INT16). 연산 경로에서 float을 사용하지 않고, 고정소수점만 사용한다.
- **Q6.10**: 활성화와 Conv 출력의 고정소수점 형식. 부호 1비트, 정수 5비트, 소수 10비트. 값 = `int16_t / 1024`.
- **가중치 스케일(Scale_W)**: INT8 가중치 텐서별 하나. `w_f32 = w_int8 * Scale_W`. `weights_w8.bin`에 텐서마다 저장되어 로더가 반환한다.
- **Multiplier (Q0.16)**: Requant 시 사용. `multiplier = round(Scale_W * 65536)`. 누산기(int32)에 곱한 뒤 16비트 우측 시프트로 Q6.10 출력을 만든다.
- **Bias**: float bias를 Conv 누산기와 같은 스케일로 맞춘 정수 `Bias_q = round(Bias_f * 1024 / Scale_W)`. int32로 저장하며, **Requant 전에** 누산기에 더한다.
- **Detect 출력 → float**: Conv 내부에서 이미 multiplier로 스케일이 적용된 Q6.10이므로, float 변환은 `값 / 1024.0f` 만 수행한다. 가중치 Scale_W를 다시 곱하면 이중 스케일이 되어 잘못된 결과가 난다.

---

## 2. 데이터 형식 및 타입

### 2.1 `csrc/types_w8a16.h`

- **Q6.10**
  - `Q6_10_FRAC_BITS = 10`, `Q6_10_SCALE = 1024`
  - 표현 범위: 약 [-32, 32)
- **타입**
  - `activation_t_w8a16` = `int16_t` (입력/활성화)
  - `weight_t_w8a16` = `int8_t` (가중치)
  - `accum_t_w8a16` = `int32_t` (Conv 누산기)
- **헬퍼**
  - `float_to_q610(float x)`: `x * 1024` 후 clamp to int16_t
  - `q610_to_float(int16_t x)`: `(float)x / 1024.0f`

### 2.2 Multiplier (Scale_W → Q0.16)

`main.c` 및 C3/SPPF/Detect 내부에서 사용:

```c
static inline uint32_t scale_to_mult(float s) {
    if (s <= 0.f) return 1U;
    uint32_t u = (uint32_t)(s * 65536.0f + 0.5f);
    return (u < 1) ? 1U : u;
}
```

- 입력 `s` = 로더에서 받은 **Scale_W**.
- 출력 = Requant 식 `(acc * multiplier + 32768) >> 16` 에 쓰는 정수.

### 2.3 Bias 변환

float bias → int32 (누산기와 동일 스케일):

```c
static void w8a16_bias_convert(const float* b, float scale, int c_out, int32_t* out) {
    if (!b || scale <= 0.f) { for (int k = 0; k < c_out; k++) out[k] = 0; return; }
    float factor = 1024.0f / scale;
    for (int k = 0; k < c_out; k++)
        out[k] = (int32_t)roundf(b[k] * factor);
}
```

- `scale`: 해당 Conv의 **Scale_W** (가중치와 동일 텐서).
- 공식: `Bias_q = round(Bias_f * 1024 / Scale_W)`.

---

## 3. 가중치 로더

### 3.1 `csrc/utils/weights_loader.h` / `.c`

- **weights_w8.bin**: INT8 텐서 + 텐서별 `scale`(Scale_W) 등 메타데이터. 로더가 파싱하여 `tensor_info_t` 배열로 보관.
- **weights_get_tensor_data(loader, name)**: FP32 텐서(bias 등) 포인터 반환. INT8 텐서에는 사용하지 않음.
- **weights_get_tensor_for_conv(loader, name, out_scale, out_is_int8)**:
  - INT8 conv 가중치용. 반환값: 가중치 포인터(void*).
  - `out_scale`에 해당 텐서의 **Scale_W** 저장.
  - `out_is_int8`에 1 저장.
  - W8A16에서는 이 Scale_W로 `scale_to_mult()`와 `w8a16_bias_convert()`에 사용.
- **weights_find_tensor(loader, name)**: `tensor_info_t*` 반환. `shape[0]` 등으로 출력 채널 수 조회 가능.

### 3.2 4-way 가중치 패킹 (Repack)

W8A16 Conv 가중치는 로드 직후 **모든** 4D INT8 텐서에 대해 repack을 적용한다.

- **레이아웃**: `[OC, IC, KH, KW]` → `[OC_padded/4, IC, KH, KW]`. 같은 `(ic, kh, kw)`에 대한 **연속된 OC 4개**를 하나의 `uint32_t`로 묶어 저장(4 int8 per uint32_t).
- **OC가 4의 배수가 아닌 경우**(예: Detect 헤드 255채널): `OC_padded = (OC + 3) & ~3`으로 올린 뒤, 실제 OC 밖의 슬롯은 **0으로 패딩**. 이렇게 하면 `conv2d_nchw_w8a16`이 어떤 레이어든 항상 `uint32_t` 단위로만 읽을 수 있다.
- **버퍼 크기**: `(OC_padded * IC * KH * KW)` 바이트. repack 후 `data_int8`는 이 새 버퍼를 가리키며, `data_owned = 1`로 해제 책임은 로더가 진다.
- **4바이트 정렬**: 패킹 버퍼는 **반드시 4바이트 정렬**된 주소에 할당한다. C11에서는 `aligned_alloc(4, size)` 사용, 그 외에는 `malloc(size)`(호스트는 보통 8바이트 정렬). **BARE_METAL(MicroBlaze 등)에서는** 비정렬 `uint32_t` 접근 시 하드웨어 에러가 날 수 있으므로, **4바이트 정렬을 보장하는 전용 할당자**(또는 `aligned_alloc`)를 사용할 것을 권장한다.

---

## 4. 저수준 연산 (operations)

### 4.1 Conv2D — `csrc/operations/conv2d_w8a16.c`, `.h`

- **시그니처**
  - 입력: `int16_t* x` (Q6.10), `int8_t* w`(실제로는 **packed 버퍼**, 아래 참조), `int32_t* bias_or_null`, `uint32_t multiplier`
  - 출력: `int16_t* y` (Q6.10)
- **가중치 레이아웃 (packed)**
  - 로더가 넘기는 `w`는 §3.2 repack된 버퍼. `(const uint32_t*)w`로 해석하여 **한 번에 4개 OC**를 읽는다.
  - **1x1**: `w_p[(oc0/4 + b4/4) * packed_oc_stride + ic * packed_ic_stride]`. `packed_oc_stride = c_in`, `packed_ic_stride = 1`.
  - **일반(KH,KW)**: `w_p[og * packed_oc_stride + ic * packed_ic_stride + kh*k_w + kw]`. `packed_oc_stride = c_in * k_h * k_w`, `packed_ic_stride = k_h * k_w`.
  - 하나의 `uint32_t`를 로드한 뒤 4바이트를 unpack하여 `w0..w3`로 쓰고, 하나의 `x_val`에 대해 `acc[b4..b4+3]`에 각각 `x_val*w0`, `x_val*w1`, … 누적. **OC가 4의 배수가 아닌 경우**(예: Detect 255) 패딩된 슬롯은 `if (b4+j < n_oc)`로 누적에서 제외한다.
- **내부**
  - 누산기: `int32_t`. `acc = bias_or_null[oc]` 초기화 후 위 packed 4-way 누적.
  - **Bias는 multiplier 연산 전에** 더해짐.
  - Requant: `out = clamp_s16((acc * multiplier + 32768) >> 16)`.
- **1x1 / 일반 커널** 동일 requant 공식. 타일링·경계 처리 등은 conv2d_w8a32와 동일한 safe 영역 사용.
- FP32 입력 경로(`conv2d_nchw_f32_w8a16`, `conv2d_nchw_f32_w8_w8a16`)는 W8A32 호환용으로만 존재하며, W8A16 추론 메인 경로에서는 사용하지 않음.

### 4.2 SiLU — `csrc/operations/silu_w8a16.c`, `silu_lut_data.h`

- **정수 경로**: `silu_nchw_w8a16(int16_t* x, ..., int16_t* y)`
  - 픽셀당 `y[i] = silu_lut_q610[(uint16_t)x[i]]`.
  - `silu_lut_q610[]`: 65536개 int16_t. 인덱스를 int16_t로 해석한 값 `v`에 대해 `x = v/1024`, `SiLU(x) = x*sigmoid(x)`를 Q6.10으로 반올림한 값.
- **LUT 생성**: `tools/gen_silu_lut.py` → `csrc/operations/silu_lut_data.h` 생성. 프로젝트에 이미 포함된 헤더만 사용하면 됨.

### 4.3 Concat — `csrc/operations/concat_w8a16.c`

- `concat_nchw_w8a16`, `concat4_nchw_w8a16`: 입출력 모두 int16_t (Q6.10). 채널 방향으로 이어 붙임. 스케일 변경 없음.

### 4.4 MaxPool2D — `csrc/operations/maxpool2d_w8a16.c`

- 입출력 int16_t (Q6.10). 최댓값만 선택하므로 스케일 유지.

### 4.5 Upsample — `csrc/operations/upsample_w8a16.c`

- Nearest 2×. 입출력 int16_t (Q6.10). 스케일 유지.

### 4.6 Bottleneck — `csrc/operations/bottleneck_w8a16.c`

- `Conv1x1 → SiLU → Conv3x3 → SiLU` (+ shortcut). 입출력 int16_t.
- 인자: 가중치 포인터, 채널 수, bias(int32), multiplier(uint32_t) 각 2세트. C3 블록 **내부**에서만 호출되며, C3가 로더에서 Scale_W/mult/bias를 계산해 넘긴다.

---

## 5. 고수준 블록 (blocks)

### 5.1 Conv 블록 — `csrc/blocks/conv_w8a16.c`, `.h`

- **conv_block_nchw_w8a16**: Conv2D 한 번 + SiLU. 입출력 int16_t.
- 인자: `(int8_t* w, c_out, k_h, k_w, int32_t* bias_or_null, uint32_t multiplier, stride, pad, ...)`.
- **main.c**에서만 호출. 매 레이어마다 `weights_get_tensor_for_conv(weights, "model.N.conv.weight", &s, &i8)`로 가중치와 Scale_W를 받고, `w8a16_bias_convert(bias_f, s, c_out, bias_buf)`, `scale_to_mult(s)`로 mult를 구한 뒤 `conv_block_nchw_w8a16(..., bias_buf, scale_to_mult(s), ...)` 호출.

### 5.2 C3 — `csrc/blocks/c3_w8a16.c`, `.h`

- **API**: `c3_nchw_w8a16(weights_loader_t* loader, int16_t* x, n, c_in, h, w, cv1_weight_name, cv2_weight_name, cv3_weight_name, n_bottleneck, bn_cv1_weight_names, bn_cv2_weight_names, shortcut, int16_t* y)`.
- **역할**: 로더와 가중치 이름만 받고, **블록 내부**에서
  - cv1/cv2/cv3: `weights_get_tensor_for_conv`로 가중치·Scale_W 로드 → `scale_to_mult`로 mult 계산 → `weights_get_tensor_data`로 bias 로드 → `weight_name_to_bias_name`으로 bias 이름 유도 후 `bias_convert` → conv+silu 호출.
  - Bottleneck 각 단계: 동일하게 로더에서 해당 가중치 이름으로 Scale_W/mult/bias 계산 후 `bottleneck_nchw_w8a16` 호출.
- **채널 수**: `weights_find_tensor(loader, weight_name)->shape[0]`로 조회.
- `weight_name_to_bias_name`: `"*.weight"` → `"*.bias"` 문자열 치환.

### 5.3 SPPF — `csrc/blocks/sppf_w8a16.c`, `.h`

- **API**: `sppf_nchw_w8a16(weights_loader_t* loader, int16_t* x, n, c_in, h, w, cv1_weight_name, cv2_weight_name, pool_k, int16_t* y)`.
- **내부**: cv1/cv2에 대해 로더로 가중치·Scale_W·bias 로드 후 mult·bias 변환. Conv1x1→SiLU, MaxPool×3, Concat4, Conv→SiLU 순서의 코어 함수 호출.
- **main.c**: `sppf_nchw_w8a16(weights, l8, n, 256, 20, 20, "model.9.cv1.conv.weight", "model.9.cv2.conv.weight", 5, l9)` 한 줄로 호출.

### 5.4 Detect — `csrc/blocks/detect_w8a16.c`, `.h`

- **API**: `detect_nchw_w8a16(weights_loader_t* loader, p3, p4, p5 (각 int16_t, 채널·해상도), m0_weight_name, m1_weight_name, m2_weight_name, c_detect, p3_out, p4_out, p5_out)`.
- **내부**: m0/m1/m2 각각 로더에서 가중치·Scale_W·bias 로드 → mult·bias 변환 → 1×1 Conv 3회 (SiLU 없음). 출력은 Q6.10 int16_t.
- **main.c**: Detect 호출 후 `p3_out[i] = (float)p3_i16[i] / 1024.0f` (p4, p5 동일). **s0/s1/s2(Scale_W) 곱하지 않음.**

---

## 6. main.c — W8A16 추론 흐름

- **조건**: `#ifdef USE_W8A16` 구간. `yolov5n_inference_w8a16(img, weights, p3_out, p4_out, p5_out, &cy_backbone, &cy_neck, &cy_head)`.
- **입력 양자화**: 전처리된 float 이미지 `img->data[]`를 `v = img->data[i] * 1024` 후 clamp하여 `int16_t x0[]`에 저장 (Q6.10).
- **메모리**: 모든 중간 피처맵은 `feature_pool_scratch_alloc`으로 할당. `feature_pool_scratch_reset()`을 추론 시작 시 한 번 호출.
- **레이어 순서** (요약):
  - L0: Conv 6×6 s2 → l0
  - L1: Conv 3×3 s2 → l1
  - L2–L8: C3 / Conv / C3 / Conv / C3 / Conv / C3 → l2…l8
  - L9: SPPF(l8) → l9
  - L10: Conv 1×1 → l10
  - L11: Upsample(l10) → l11
  - L12: Concat(l11, l6) → l12
  - L13: C3 → l13
  - L14: Conv 1×1 → l14
  - L15: Upsample(l14) → l15
  - L16: Concat(l15, l4) → l16
  - L17: C3 → l17 (P3)
  - L18: Conv 3×3 s2 → l18
  - L19: Concat(l18, l14) → l19
  - L20: C3 → l20 (P4)
  - L21: Conv 3×3 s2 → l21
  - L22: Concat(l21, l10) → l22
  - L23: C3 → l23 (P5)
  - L24: Detect(l17, l20, l23) → p3_i16, p4_i16, p5_i16
- **Conv 블록 호출 패턴** (L0, L1, L3~L8, L10, L14, L18, L21 등):  
  `W_CONV_W16("model.N.conv.weight", &s, &i8)`, `W_W16("model.N.conv.bias")`, `w8a16_bias_convert(b, s, c_out, bias_buf)`, `conv_block_nchw_w8a16(..., bias_buf, scale_to_mult(s), ...)`.
- **C3/SPPF/Detect**: 위와 같이 로더 + 텐서 이름만 전달. 블록 내부에서 Scale_W·mult·bias 계산.
- **Detect 이후**: `p3_out[i] = (float)p3_i16[i] / 1024.0f` (p4, p5 동일). 이 float 배열이 기존 Decode/NMS로 전달된다.
- **타이밍**: 각 레이어 전후 `yolo_timing_set_layer`, `timer_read64`/`timer_delta64`, `LAYER_LOG_VAL`, `yolo_timing_print_layer_ops`. Conv 블록·SiLU 등은 블록 내부에서 `yolo_timing_begin("conv2d")`/`yolo_timing_end()` 등으로 구간 기록.

---

## 7. 공통 유틸

- **feature_pool** (`csrc/utils/feature_pool.c,h`): W8A16에서는 `feature_pool_scratch_reset()` 후 `feature_pool_scratch_alloc()`만 사용. free 없이 한 번의 추론 동안 연속 할당.
- **decode.c, nms.c**: W8A32/W8A16 공통. 입력은 항상 float (W8A16은 Detect 출력을 /1024로 float 변환한 뒤 전달).
- **timing.c,h, mcycle.h**: 레이어·연산별 시간 측정 및 출력. W8A32와 동일한 구간 이름·ms 단위 사용.

---

## 8. 빌드

- **정의**: `-DUSE_W8A16 -DUSE_WEIGHTS_W8`
- **소스**:  
  `main.c` + blocks: `conv_w8a16.c c3_w8a16.c decode.c detect_w8a16.c nms.c sppf_w8a16.c`  
  + operations: `bottleneck_w8a16.c concat_w8a16.c conv2d_w8a16.c maxpool2d_w8a16.c silu_w8a16.c upsample_w8a16.c`  
  + utils: `feature_pool.c image_loader.c weights_loader.c timing.c uart_dump.c`
- **가중치 파일**: `assets/weights_w8.bin` (INT8 + per-tensor scale).
- **실행**: `./run_compare_host.sh w8a16` 또는 위 소스/옵션으로 직접 gcc.

---

## 9. 파일별 역할 요약

| 파일 | 역할 |
|------|------|
| `types_w8a16.h` | Q6.10 상수·타입·float↔Q6.10 변환 |
| `main.c` (USE_W8A16) | scale_to_mult, w8a16_bias_convert, 입력 Q6.10 변환, L0–L24 호출, Detect 출력 /1024 → float |
| `weights_loader.c,h` | weights_w8.bin 로드, get_tensor_for_conv(Scale_W 반환), get_tensor_data(bias). **4D INT8 repack**: [OC,IC,KH,KW]→[OC_padded/4,IC,KH,KW] uint32_t, 0 패딩, 4바이트 정렬 할당. |
| `conv2d_w8a16.c,h` | int16 in/out, **packed** w(uint32_t 단위 4-way 로드·누산), int32 acc+bias, (acc*mult+32768)>>16 → int16 |
| `silu_w8a16.c` + `silu_lut_data.h` | LUT 기반 SiLU Q6.10→Q6.10 |
| `conv_w8a16.c,h` | conv2d + silu, multiplier/bias 인자 전달 |
| `c3_w8a16.c,h` | 로더+이름 API, 내부 Scale_W/mult/bias 계산, cv1/cv2/cv3 + bottleneck |
| `sppf_w8a16.c,h` | 로더+이름 API, 내부 Scale_W/mult/bias, cv1–maxpool–concat–cv2 |
| `detect_w8a16.c,h` | 로더+이름 API, 내부 Scale_W/mult/bias, 3× 1×1 Conv |
| `bottleneck_w8a16.c,h` | Conv1×1–SiLU–Conv3×3–SiLU (+shortcut), C3에서만 호출 |
| `concat_w8a16.c,h`, `maxpool2d_w8a16.c,h`, `upsample_w8a16.c,h` | Q6.10 유지 연산 |
| `gen_silu_lut.py` | SiLU LUT C 헤더 생성 |

이 문서는 위 구조와 수식에 맞춰 현재 코드베이스가 어떻게 동작하는지만 기술한 것이다.
