# W8A32 Mixed-Precision 계획 (YOLOv5n)

## 목표 하드웨어
- **Xilinx Arty A7-35T**, MicroBlaze V Soft-core
- **병목**: 전체 실행 시간의 ~72%가 DDR3 가중치 로딩으로 인한 Memory Stall

## 전략: 가중치만 INT8, 연산은 FP32 유지 (W8A32)

- **가중치 INT8**: DDR 전송량을 1/4로 축소 → 메모리 스톨 감소 기대.
- **연산 FP32 유지**: C 측에서 INT8 가중치를 레이어별 scale로 디양자화한 뒤 기존 FP32 conv 그대로 사용. DSP는 유휴 시 FP 연산으로 활용 가능.
- **Bias / 기타 파라미터**: FP32 유지 (양자화하지 않음).

## 현재 코드 구조 요약

### weights.bin (FP32) 포맷
- `num_tensors` (4B)
- 텐서별: `key_len`(4) → `key`(UTF-8) → `ndim`(4) → `shape[]`(4×ndim) → **4B 정렬** → `float32[]` (num_elements×4)
- C: `weights_loader.c`가 동일 포맷 파싱, `weights_get_tensor_data(loader, "model.0.conv.weight")` 등으로 접근.

### preprocessed_image.bin 포맷
- **헤더 24B (반드시 유지)**:
  - `original_w`(4), `original_h`(4), `scale`(4), `paste_x`(4), `paste_y`(4), `size`(4)
- 이후: `3 × size × size × sizeof(float)` NCHW 이미지 데이터.
- C: `image_loader.c`가 헤더 24B 파싱 후 `img->data`를 헤더 바로 다음부터 사용. `platform_config.h`의 `IMAGE_HEADER_SIZE == 24`.

### 레이어별 가중치 사용 (main.c)
- **Conv 블록**: `model.{0,1,3,5,7,10,14,18,21}.conv.weight` / `.bias`
- **C3**: `model.{2,4,6,8,13,17,20,23}.cv1/cv2/cv3.conv.weight` + `model.{2,4,6,8,...}.m.*.cv1/cv2.conv.weight`
- **SPPF**: `model.9.cv1/cv2.conv.weight`
- **Detect**: `model.24.m.{0,1,2}.weight` / `.bias`
- `.weight`만 INT8로 양자화; `.bias`는 FP32 유지.

---

## 할 일 정리

### 1. quantize_weights.py (구현 완료)
- **입력**: 기존 `weights.bin` (FP32).
- **동작**:
  - 레이어(텐서)별 **Symmetric Quantization**:
    - 가중치 텐서만 대상 (키가 `.weight`로 끝나는 것).
    - `scale = max(|w|) / 127` (max가 0이면 작은 epsilon 사용).
    - `w_int8 = round(w_f32 / scale)`, clamp to `[-127, 127]`.
  - **출력**: `assets/weights_w8.bin` (scale은 w8 내부 텐서 헤더에 포함). `--out-scales` 시 `scales.bin` 선택 출력(호환용).

**weights_w8.bin 포맷 (A: 텐서별 scale 포함 → 순서/일부만 INT8 변경에도 안전)**  
- `num_tensors` (4B, little-endian)  
- 텐서별: `key_len`(4) → `key`(UTF-8) → `ndim`(4) → `shape[]`(4×ndim) → `dtype`(1B, 0=float32, 1=int8)  
  - dtype==INT8일 때: **scale**(4B float) → **4B 정렬 패딩** → int8 데이터 (num_elems×1B)  
  - dtype==FLOAT32일 때: **4B 정렬 패딩** → float32 데이터 (num_elems×4B)  
- **D: 데이터 정렬**: float 데이터 시작·`dequant_buf`(malloc)는 4B 정렬 유지.

### 2. C 측 변경 (W8A32) — 구현 완료
- **weights_loader**: `weights_w8.bin`만 로드 (scale은 w8 내부).
  - `weights_load_from_file_w8(w8_path)` (호스트), `weights_init_from_memory_w8(w8_base, w8_size)` (BARE_METAL).
  - `weights_get_tensor_for_conv(loader, name, &scale, &is_int8)`로 conv용 `int8_t*`+scale 반환.
  - **dequant 풀 제거** (보드 heap 4MB 한계). 모든 Conv는 on-the-fly 디양자화.
- **conv2d**: `conv2d_nchw_f32_w8` — local_w pre-load, 32비트 번들 로드, 1×1 fast path.
- **conv_block**: `(void* w, float w_scale, int w_is_int8)` 받아 W8이면 `conv2d_nchw_f32_w8` 호출.
- **C3**: `c3_nchw_f32`가 cv1/cv2/cv3 및 bottleneck 내부 cv1/cv2에 대해 `(void*, scale, is_int8)` 수신. 내부 `conv1x1`·`bottleneck_nchw_f32`가 W8 분기.
- **Detect**: `detect_nchw_f32`가 m0/m1/m2 가중치에 대해 `(void*, scale, is_int8)` 수신, 1×1 conv 세 번 각각 W8/FP32 분기.
- **빌드**: `-DUSE_WEIGHTS_W8` 시 W8 가중치 사용.  
  호스트: `assets/weights_w8.bin`  
  BARE_METAL: DDR에 `weights_w8.bin` → `WEIGHTS_W8_DDR_BASE` (`platform_config.h`).

**B (보드 메모리):** dequant 풀 제거. (ic,b)당 local_w[36]에 int8→float 변환 후 FP32 연산. 32비트 번들 로드, 1×1 fast path 적용. 자세한 내용은 `W8A32_IMPLEMENTATION.md` 참고.

### 3. preprocessed_image.bin (processed_image.bin)
- **현재**: 헤더 24B + FP32 픽셀 데이터. C는 `IMAGE_HEADER_SIZE`로 헤더를 건너뛰고 데이터만 사용.
- **진행 시 유의**: 이미지도 INT8로 줄이려면 **헤더 24B는 그대로 두고**, 그 뒤 페이로드만 `int8` + (전역 또는 채널별) scale로 바꾸는 식으로 설계해야 함. 포맷 변경 시 `image_loader.c`와 `platform_config.h`의 `IMAGE_DATA_SIZE` 등과 일치시킬 것.

---

## Symmetric Quantization 수식 (참고)

- **양자화**: `scale = max(|W|) / 127`, `q = round(W / scale)`, `q ∈ [-127, 127]`.
- **디양자화 (C: W8A32 핵심)**  
  - 로더에서 버퍼에 채울 때: `dst[i] = (float)src[i] * scale;`  
  - conv 루프 내 인라인(보드용 대안): `contrib += x[idx] * ((float)w_int8[w_idx] * scale);`
