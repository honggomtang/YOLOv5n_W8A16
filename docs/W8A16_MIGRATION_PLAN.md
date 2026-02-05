# W8A16 마이그레이션 계획

## 현재 상태 (W8A32)
- **conv2d**: `csrc/operations/conv2d.c` — `conv2d_nchw_f32`, `conv2d_nchw_f32_w8` (가중치 int8 + scale, 활성화 float)
- **silu**: `csrc/operations/silu.c` — `silu_nchw_f32` (float in/out, `x * sigmoid(x)`)
- **데이터 흐름**: 입력/중간/출력 모두 `float*`. conv2d_w8은 int8 가중치를 루프 내에서 `(float)w_int8*scale`로 복원 후 float MAC.

## 목표 (W8A16)
- **Activation**: float → **int16_t** 고정소수점 (Q6.10: 부호 1비트, 정수 5비트, 소수 10비트)
- **Weight**: int8_t 유지 (연산 시 16-bit 정수로 취급)
- **Accumulator**: int32_t
- **Requantization**: 곱셈 후 scale 대신 **비트 시프트(>>)** 로 int16_t 범위 맞춤
- **float 사용 금지** (순수 정수 연산)

## 디렉터리/파일 구조 (변경 후)

```
csrc/
├── types_w8a32.h   # float 기반 타입 (기존 동작)
├── types_w8a16.h   # Q6.10, int16_t activation, int32_t acc
├── operations/
│   ├── conv2d_w8a32.c / .h
│   ├── conv2d_w8a16.c / .h
│   ├── silu_w8a32.c / .h
│   ├── silu_w8a16.c / .h
│   ├── bottleneck_w8a32.c / .h
│   ├── bottleneck_w8a16.c / .h
│   ├── concat_w8a32.c / .h
│   ├── concat_w8a16.c / .h
│   ├── maxpool2d_w8a32.c / .h
│   ├── maxpool2d_w8a16.c / .h
│   ├── upsample_w8a32.c / .h
│   └── upsample_w8a16.c / .h
├── blocks/
│   ├── conv_w8a32.c / .h
│   ├── conv_w8a16.c / .h
│   ├── c3_w8a32.c / .h
│   ├── c3_w8a16.c / .h
│   ├── sppf_w8a32.c / .h
│   ├── sppf_w8a16.c / .h
│   ├── decode.c / .h   (출력 단 decode는 유지 또는 별도 처리)
│   ├── detect_w8a32.c / .h
│   ├── detect_w8a16.c / .h
│   ├── nms.c / .h
│   └── ...
```

## 구현 규칙 (W8A16)

| 항목 | 규칙 |
|------|------|
| 포맷 | Q6.10 (1부호 + 5정수 + 10소수) |
| 입력/활성화 | float → int16_t (저장 시 값×2^10) |
| 가중치 | int8_t 유지, 연산 시 16-bit 정수로 취급 |
| 누산기 | 곱셈 결과 16×8=24비트 → int32_t 누산 |
| Requantization | 결과를 적절한 시프트(>>)로 int16_t 범위로 복원 |

## 작업 순서

1. **파일/함수 이름 변경**  
   - operations/, blocks/ 내 기존 .c/.h를 `*_w8a32.c`/`*_w8a32.h`로 변경  
   - 모든 public 함수 이름 뒤에 `_w8a32` 접미사 추가  

2. **_w8a16 복사본 생성**  
   - 각 `*_w8a32.*`를 `*_w8a16.*`로 복사  
   - 앞으로 `*_w8a16` 파일만 수정하여 고정소수점 구현  

3. **타입 헤더 이원화**  
   - `types_w8a32.h`: float 기반 (기존)  
   - `types_w8a16.h`: int16_t activation, int32_t acc, Q6.10 관련 매크로/타입  

4. **연산별 W8A16 구현**  
   - conv2d: `conv2d_nchw_w8a16` — 정수 곱셈 + 시프트, 1x1 fast path / bundle loading 유지  
   - silu: LUT 기반 정수 SiLU  
   - concat, maxpool2d, upsample: int16_t 입출력 버전  
   - bottleneck, c3, sppf, conv_block: int16_t 파이프라인  

5. **비교 검증**  
   - W8A32 vs W8A16 동일 입력에 대해 출력 비교 (허용 오차 또는 스케일 맞춘 후 비교)

## SiLU 정수화 (LUT)

- SiLU(x) = x * sigmoid(x). **Q6.10 → Q6.10** LUT: 입력 v는 v/1024로 해석, 출력 = round(SiLU(x)*1024). (정밀도 유지)
- LUT 65,536 엔트리. 인덱싱: `(uint16_t)(int16_t)x` 로 2의 보수 비트 패턴 그대로 사용.

## 레이어별 shift_amount / Q6.10 오버플로우 대책

- Q6.10 표현 범위는 **±31.999** (정수부 5비트). Conv 결과(sum)가 이보다 크면 오버플로우.
- **L1** 등 초반 레이어는 sum이 46 등으로 31.9를 넘을 수 있음 → **shift_amount를 12** 등으로 키워 결과를 Q6.10 범위 안으로 눌러줌 (예: 46 → acc>>12 ≈ 11.5).
- 레이어별로 `shift_amount`를 calibration 하거나, Q6.10 포맷이 적절한지 검토 필요.

---

## 지금까지 완료한 작업 요약 (상세)

### 1. 프로젝트 구조 및 타입 분리 (초기 마이그레이션)

- **파일/함수 이름**: `operations/`, `blocks/` 내 모든 소스를 `*_w8a32.c`/`*_w8a32.h`로 통일하고, 동일 내용의 `*_w8a16.c`/`*_w8a16.h` 복사본 생성.
- **타입 헤더**: `types_w8a32.h`(float 기반), `types_w8a16.h`(Q6.10, int16_t 활성화) 분리.
- **main/빌드**: W8A32 소스를 참조하도록 설정. W8A16은 별도 경로에서 점진 구현.

---

### 2. Conv2D W8A16 (정수 전용 경로)

**목적**: float 없이 int16_t 입출력, int8_t 가중치, int32_t 누산으로 conv 구현.

- **함수**: `conv2d_nchw_w8a16(..., int32_t shift_amount, ...)`  
  - 입력 `x`: int16_t (Q6.10, 값×1024).  
  - 가중치 `w`: int8_t **그대로** 사용 (W8A32처럼 `local_w`에 scale 곱하지 않음).  
  - 누산: `acc += (int32_t)x_val * (int32_t)w_val` (int32_t).  
  - 출력: `y = clamp_s16((acc + round_bias) >> shift_amount)`.

- **반올림 시프트 (Rounding Shift)**  
  - `round_bias = (shift_amount > 0) ? (1 << (shift_amount - 1)) : 0`  
  - `(acc + round_bias) >> shift_amount` 로 **floor 대신 nearest** 적용 → 한쪽으로 쏠리는 드리프트 감소.

- **Bias 단위 (헤더 주석)**  
  - `bias_or_null`는 int32 단위. float bias 사용 시  
    `Bias_int32 = round(Bias_float * (2^shift_amount / Scale_weight))` 로 변환해야 함.  
  - `(int32_t)bias_float` 그대로 넣으면 약 1024배 작게 들어가 오차 발생.

- **shift_amount vs weight_scale (헤더 주석)**  
  - W8A32의 `scale`은 부동소수점(예: 0.091359).  
  - W8A16의 `shift_amount`는 2^shift(예: 10 → 1024)로 requant에만 사용.  
  - scale ≠ 1/1024 이면 `conv_f32/scale`과 `acc>>10` 사이에 미세 차이는 **양자화 차이**일 뿐 로직 오류 아님.

- **패딩/경계**: W8A32와 동일한 `safe_oh_min/max`, `safe_ow_min/max`, 타일/픽셀별 `in_safe` 분기 사용.

---

### 3. Conv2D vs W8A32 검증 및 디버깅

- **테스트**: `tests/test_conv2d_w8a16_compare.c`  
  - 동일 float 입력을 Q6.10으로 변환해 W8A16에 넣고, W8A32 출력/scale과 W8A16 int16 출력을 비교.

- **trace_pixel_diff()**  
  - 문제 픽셀 (n,c,h,w)에 대해 W8A32/W8A16이 참조하는 입력 (ih, iw, x_val, w_val)과  
    `in_safe`, `x_base`(ih0, iw0) 계산식을 stderr로 출력.  
  - 결과: 해당 픽셀에서 **in_safe·x_base·참조 윈도우가 양쪽 동일**함을 확인.  
  - 206 수준의 max diff는 **패딩/인덱스 버그가 아니라**, float vs Q6.10 양자화·반올림 차이로 결론.

---

### 4. SiLU W8A16 (LUT)

**목적**: int16_t(Q6.10) 입출력, float 연산 없이 SiLU(x)=x*sigmoid(x) 적용.

- **LUT 규격**  
  - 크기 **65,536** (int16_t 전체).  
  - **Q6.10 → Q6.10**: 입력 인덱스 `i`에 대응하는 값 `v = (int16_t)(uint16_t)i`를 **x = v/1024** 로 해석.  
  - `silu = x * sigmoid(x)`, **출력 = round(silu * 1024)** 후 int16 클램프.  
  - 이렇게 해야 고정소수점 정밀도를 유지 (정수만 쓰는 Q15.0 방식이 아님).

- **음수 인덱싱**  
  - `idx = (uint16_t)x[i]`: int16_t 비트 패턴 그대로 사용.  
  - -1 → 65535, -32768 → 32768. LUT 생성 시에도 `i=0..65535` → `v=(int16_t)(uint16_t)i` 로 2의 보수 일치.

- **블록**: `conv_block_nchw_w8a16` = `conv2d_nchw_w8a16` → `silu_nchw_w8a16` (in-place).

---

### 5. Conv + SiLU 파이프라인 검증 (Q6.10 복원)

- **L1 오버플로우 대책**  
  - Q6.10 표현 범위 ±31.999. L1 conv 결과(sum)가 46 등으로 넘을 수 있음.  
  - **L1에서는 shift_amount=12** 사용 → acc>>12 로 결과를 Q6.10 범위 안(예: ~11.5)으로 유지.

- **검증 방식** (`tests/test_conv_silu_w8a16_compare.c`)  
  - W8A16만 사용: conv(shift=12) → conv_out (Q6.10) → SiLU LUT → y_w8a16.  
  - 참조: `ref_float = SiLU(conv_out/1024)`, `ref_q = round(ref_float*1024)`.  
  - **y_w8a16과 ref_q(Q6.10 양자화 참조)를 비교** → Max diff 0으로 통과.

- **문서**: `W8A16_MIGRATION_PLAN.md`에 레이어별 shift_amount·Q6.10 오버플로우 대책 정리.

---

### 6. 원자 연산 (Atomic Operations) — 입출력 int16_t

모두 **스케일 변화 없이** 데이터 타입만 int16_t(Q6.10)로 통일.

| 연산 | 파일 | 함수 | 설명 |
|------|------|------|------|
| **MaxPool2D** | maxpool2d_w8a16.c/.h | `maxpool2d_nchw_w8a16` | 최댓값만 선택, 스케일 동일. 검증: test_maxpool2d_w8a16_compare.c |
| **Upsample 2x** | upsample_w8a16.c/.h | `upsample_nearest2x_nchw_w8a16` | nearest 2x, 값 복사만. 검증: test_upsample_w8a16_compare.c |
| **Concat** | concat_w8a16.c/.h | `concat_nchw_w8a16`, `concat4_nchw_w8a16` | 채널 방향 결합. **합쳐지는 모든 텐서가 동일 Q6.10 스케일**이어야 함(헤더/주석에 명시). 검증: test_concat_w8a16_compare.c |

---

### 7. 위험성 점검 (Feature Pool / In-place)

**① Feature Pool 파편화**

- `feature_pool_alloc`/`free`는 **First-fit + coalescing** 방식이라, C3 한 번에 Bottleneck 3개면 alloc/free가 수십 번 발생. MicroBlaze 등에서 malloc 계열은 느리고 **파편화**에 취약할 수 있음.
- **대응**: `feature_pool_scratch_reset()` / `feature_pool_scratch_alloc(size)` 스크래치패드 API 추가. 추론 시작 시 `scratch_reset()` 한 번, 추론 중에는 `scratch_alloc()`만 사용하고 free 없이 포인터만 밀어서 사용하면 파편화 없음. BARE_METAL에서는 이 모드 사용 권장.
- 기존 `alloc`/`free`는 호스트/유연성용으로 유지. 블록들을 scratch 전용으로 바꾸면 추론 한 번당 할당 횟수·파편화 위험 감소.

**② cv1_out과 y의 Aliasing (In-place)**

- `silu_nchw_w8a16(y, ..., y)` 처럼 출력 버퍼를 입력으로도 쓰는 in-place 사용 있음.
- **SiLU LUT**: 한 픽셀씩 독립 처리(`y[i] = LUT[x[i]]`)이므로 **in-place 안전**. 코드/주석에 명시해 둠.
- **주의**: 시프트·커널이 있는 연산 등은 in-place 시 읽기 전에 덮어쓰면 망가질 수 있음. 다른 연산에서 출력을 입력으로 재사용할 때는 해당 연산이 in-place 허용인지 문서/구현 확인 필요.

---

### 8. 아직 하지 않은 것 (다음 단계)

- **구조 블록**: bottleneck_w8a16, c3_w8a16, sppf_w8a16, detect_w8a16 (conv_block·maxpool·concat·upsample 조립).
- **레이어별 shift 테이블**: `layers_config.h` 등에 24개 레이어의 shift_amount 배열(Calibration Table) 정의.
- **경계 로직**: Detect 출력(int16) → float 변환 후 Decode/NMS로 넘기는 지점 정리 및 구현.
