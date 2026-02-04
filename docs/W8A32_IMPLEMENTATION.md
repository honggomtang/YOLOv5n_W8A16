# W8A32 구현 완료 내역

YOLOv5n 가중치 INT8 / 연산 FP32 (W8A32) 구현 내용 정리.

---

## 1. 전략 개요

| 항목 | 내용 |
|------|------|
| **가중치** | INT8 (레벨당 scale) |
| **연산** | FP32 유지 |
| **Bias/BN** | FP32 유지 |
| **효과** | DDR 가중치 전송량 약 1/4 (7.6MB→1.8MB) |

---

## 2. 파일 변경 사항

### 2.1 weights_loader (dequant 풀 제거)

| 파일 | 변경 |
|------|------|
| `weights_loader.c` | dequant_pool 할당/해제 제거 |
| `weights_loader.h` | `dequant_pool_base`, `dequant_buf_cap`, `dequant_pool_next` 제거 |

**이유**: 45MB dequant 풀은 보드 heap(4MB)에서 불가. 모든 Conv를 on-the-fly 처리로 전환.

### 2.2 SPPF (W_CONV 경로로 전환)

| 파일 | 변경 |
|------|------|
| `sppf.h` | 시그니처: `(void* cv1_w, float cv1_scale, int cv1_is_int8, ...)` |
| `sppf.c` | cv1/cv2에 `conv2d_nchw_f32_w8` 사용 (W8) 또는 `conv2d_nchw_f32` (FP32) |
| `main.c` | SPPF 호출 시 `W_CONV`로 가중치 전달 |

**이유**: SPPF가 기존 `W()`로 conv 가중치를 사용하면 dequant 풀이 필요해 heap 초과. W_CONV on-the-fly로 통일.

### 2.3 conv2d_nchw_f32_w8 최적화

| 최적화 | 내용 |
|--------|------|
| **local_w pre-load** | (ic,b)당 1회 int8→float 변환, (dh,dw) 루프는 FP32 연산만 |
| **32비트 번들 로드** | 정렬 시 `uint32_t` 1회로 4개 int8 로드, 메모리 접근 1/4 |
| **정렬 체크** | `(uintptr_t)w_src % 4 == 0`일 때만 32비트 로드, 아니면 바이트 fallback |
| **Remainder** | 3×3(9B): 앞 8B 32비트 2회, 마지막 1B 바이트 |
| **Loop unrolling** | 8바이트씩 2× unroll로 루프 횟수 절반 |
| **1×1 Fast Path** | kh,kw 루프 없음, ic outer로 입력 1회 로드 후 여러 oc에 재사용 |

---

## 3. conv2d_nchw_f32_w8 상세

### 3.1 Pre-load (local_w)

```
(ic, b)당 1회:
  local_w[0..k_h*k_w-1] = (float)w_base[i] * scale   // scale 포함(미리 반영)
  - 정렬 시: 32비트 번들 로드 (4개씩)
  - 미정렬: 바이트 로드
```

### 3.2 연산 루프

```
(dh, dw) 루프에서:
  contrib = sum over (kh, kw) of x * local_w   // FP32 * FP32
  acc[b] += contrib
```

### 3.3 1×1 Fast Path

- 조건: `k_h == 1 && k_w == 1`
- 루프: ic outer → (dh, dw) → b
- 입력 재사용: `x_val = x[ic, oh, ow]` 1회 로드 → n_oc개 출력 채널에 사용
- 1×1에서는 ic outer로 입력 재사용이 32비트 번들보다 유리

---

## 4. 빌드 및 실행

### 4.1 weights_w8.bin 생성

```powershell
py -3 tools/quantize_weights.py
# 또는
python tools/quantize_weights.py
```

- 입력: `assets/weights.bin` (FP32)
- 출력: `assets/weights_w8.bin` (~1.8MB)

### 4.2 호스트 빌드

```powershell
build_host.bat        # FP32
build_host.bat w8     # W8A32
```

### 4.3 BARE_METAL (Vitis)

- 컴파일: `-DBARE_METAL -DUSE_WEIGHTS_W8`
- DDR 0x88000000에 `weights_w8.bin` 로드 후 실행

---

## 5. platform_config.h

| 항목 | 값 | 비고 |
|------|-----|------|
| WEIGHTS_W8_DDR_BASE | 0x88000000 | WEIGHTS_DDR_BASE와 동일 |
| WEIGHTS_W8_DDR_SIZE | 4MB | ~1.8MB w8 수용 |
| SCALES_DDR_BASE | (없음) | scale은 w8 내부 텐서 헤더에 포함 |

---

## 6. 메모리 사용

| 항목 | FP32 | W8A32 |
|------|------|-------|
| 가중치 | ~7.6MB | ~1.8MB |
| dequant 풀 | - | 제거됨 (0) |
| local_w (conv 내부) | - | 36 float (144B, 스택) |
| Heap | - | 4MB 이내 유지 |

---

## 7. Data Cache (xparameters.h)

| 항목 | 값 |
|------|-----|
| MIG | 0x80000000 ~ 0x8fffffff |
| WEIGHTS_W8 | 0x88000000 (MIG 내) |
| D-Cache | 16KB, 16B 라인 |

0x88000000은 MIG 범위 내이므로 D-Cache 대상. Vivado Block Design에서 MicroBlaze↔MIG 연결이 캐시 경유인지 확인 권장.

---

## 8. 참고 문서

- `W8A32_PLAN.md` - 계획 및 weights_w8 포맷
- `W8_PERFORMANCE_ANALYSIS.md` - D-Cache, 바이트 접근 분석
- `VITIS_BUILD.md` - BARE_METAL 빌드
