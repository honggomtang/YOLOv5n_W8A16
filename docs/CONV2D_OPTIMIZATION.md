# conv2d 최적화 상세 설명 (개념 + 코드)

이 문서는 `conv2d_nchw_f32`에 적용된 최적화가 **개념적으로 무엇을 바꾸었고**, **코드상으로는 어디를 어떻게 수정했는지** 정리한 것이다.

---

## 0. 최초(원본) 구조 (참고용)

**루프 순서:** `n → oc → oh → ow → ic → kh → kw`  
**특징:** 한 출력 픽셀 (oh, ow)에 대해 한 출력 채널(oc)을 끝까지 누적한 뒤 다음 oc로. 인덱스는 매번 `x_idx`, `w_idx` 곱셈으로 계산. 패딩은 가장 안쪽 루프(kh, kw) 안에서 `if (ih/iw 범위 초과) continue`로 처리.

---

## 1. 타일링 (Tiling) — 출력 공간을 8×8 블록으로 나눔

### 개념
- **문제:** 출력 전체(h_out × w_out)를 한 번에 돌면, 한 픽셀에서 쓰던 입력/가중치가 다음 픽셀에서 이미 캐시에서 밀려났을 수 있음.
- **해결:** 출력을 **8×8 타일** 단위로 나눠서, 한 타일 안의 64개 픽셀을 연속으로 처리. 한 타일이 D-Cache(예: 16KB)에 들어갈 수 있어서 **같은 입력/가중치를 여러 번 재사용**하기 좋음.

### 코드상 변경
- **추가:** `CONV2D_TILE_H`, `CONV2D_TILE_W` (기본 8).
- **바깥 루프:** `oh`, `ow` 직접 순회 대신 **타일 시작점** `oh0`, `ow0`을 `tile_h`, `tile_w` 단위로 증가.
  - `for (oh0 = 0; oh0 < h_out; oh0 += tile_h)`
  - `for (ow0 = 0; ow0 < w_out; ow0 += tile_w)`
- **타일 내부:** `oh_end = min(oh0 + tile_h, h_out)`, `ow_end = min(ow0 + tile_w, w_out)`, `th = oh_end - oh0`, `tw = ow_end - ow0`.
- **실제 픽셀:** `oh = oh0 + dh`, `ow = ow0 + dw` 로 `dh ∈ [0, th)`, `dw ∈ [0, tw)` 만큼만 순회.

즉, **"출력 좌표 공간을 8×8 블록으로 쪼개서, 블록 단위로 메모리 접근을 묶었다"**가 코드상으로 한 일이다.

---

## 2. 출력 채널 블록 (OC_BLOCK) — oc를 32개씩 묶어서 처리

### 개념
- **문제:** 출력 채널이 수십~수백 개일 때, 한 타일에서 "입력 타일 + 모든 oc의 가중치"를 동시에 캐시에 올리기 어렵다.
- **해결:** 출력 채널을 **32개씩(OC_BLOCK)** 묶어서, 한 타일(8×8)에 대해 **그 32개 oc만** 먼저 다 계산한 뒤 다음 32개로. 한 블록 분의 누적값만 임시 버퍼에 두면 되므로 캐시/메모리 사용이 제한됨.

### 코드상 변경
- **추가:** `CONV2D_OC_BLOCK` (기본 32), **누적 버퍼** `conv2d_acc_buf[TILE_H][TILE_W][OC_BLOCK]` (BSS).
- **루프:** `oc` 한 번에 다 도는 대신 `oc0 = 0, OC_BLOCK, 2*OC_BLOCK, ...` 으로 블록 시작점만 순회.
  - `for (oc0 = 0; oc0 < c_out; oc0 += oc_block)`
  - 실제 개수: `n_oc = min(oc_block, c_out - oc0)`.
- **의미:** 한 (oh0, ow0) 타일·한 oc_block 구간에 대해서만 `conv2d_acc_buf[dh][dw][b]` (b = 0..n_oc-1)를 채우고, 다 채운 뒤 `y`에 한꺼번에 복사.

즉, **"출력 채널 차원을 32개 단위 블록으로 나눠서, 타일×블록 단위로 누적 버퍼를 쓰고, 그 결과만 y에 기록한다"**가 코드 변경이다.

---

## 3. 루프 순서 변경 — 가중치 재사용 극대화 (ic → b → dh → dw → kh → kw)

### 개념
- **이전(문제):** `ic → dh → dw → b → kh → kw`  
  → 한 픽셀 (dh, dw)에 대해 32개 필터(b)를 **번갈아** 계산. 즉, 필터 하나(예: 3×3)를 읽어서 **한 픽셀에 한 번** 쓰고 다음 필터로 넘어감. 32개 필터를 다 쓰면 캐시에서 밀려날 가능성이 큼.
- **개선:** `ic → b → dh → dw → kh → kw`  
  → **한 필터(b)** 에 대해, 그 필터가 커버하는 **타일 전체 64픽셀(dh,dw)** 을 먼저 다 계산. 필터 가중치를 **한 번 로드해서 64번 재사용**.

### 코드상 변경
- **순서 변경:**  
  - **Before:** `for (ic) { for (dh) { for (dw) { ... for (b) { contrib; acc[dh][dw][b] += contrib } } } }`  
  - **After:** `for (ic) { for (b) { const float* w_base = w + (oc0+b)*w_oc_stride + ic*w_ic_stride; for (dh) { for (dw) { contrib; acc_ptr[b] += contrib } } } }`
- **의미:** `w_base`는 (ic, b)가 바뀔 때만 갱신. (dh, dw) 루프 안에서는 **같은 w_base**를 64번 쓰므로, **가중치 64회 재사용**이 코드 구조로 보장된다.

---

## 4. Strength reduction — 안쪽 루프에서 곱셈 제거, 포인터 증감만 사용

### 개념
- **문제:** 가장 안쪽 루프(kw) 안에서 매번  
  `x_idx = ((ni*c_in+ic)*h_in+ih)*w_in+iw`,  
  `w_idx = (((oc*c_in+ic)*k_h)+kh)*k_w+kw`  
  같은 **인덱스 곱셈**을 하면, MicroBlaze처럼 곱셈이 비싼 코어에서는 "실제 곱셈(acc += x*w)"보다 **주소 계산**에 사이클을 더 쓰게 됨.
- **해결:**  
  - **kw 루프:** `x_row`, `w_row` 포인터를 두고 `contrib += (*x_row++) * (*w_row++)` 만 수행. 즉, **곱셈은 1회(실제 연산), 나머지는 포인터 +1**.  
  - **kh 루프:** `x_row = x_base + kh * x_h_stride`, `w_row = w_base + kh * w_k_stride` 로 **루프당 1번**만 stride 곱셈.

### 코드상 변경
- **Stride 상수 (함수 상단):**  
  `x_h_stride = w_in`, `x_c_stride = h_in*w_in`,  
  `w_k_stride = k_w`, `w_ic_stride = k_h*k_w`, `w_oc_stride = c_in*k_h*k_w`.
- **빠른 경로 내부 (kh, kw):**  
  - 인덱스 계산/배열 접근 제거.  
  - `x_base` = (ni, ic, oh, ow)에 대응하는 입력 시작 주소.  
  - `for (kh)` 안에서 `x_row = x_base + kh*x_h_stride`, `w_row = w_base + kh*w_k_stride`.  
  - `for (kw)` 안에서는 `contrib += (*x_row++) * (*w_row++);` 만 사용.

즉, **"가장 안쪽 루프에서는 인덱스 곱셈 없이 포인터 증감만 하도록 바꿨다"**가 코드 변경이다.

---

## 5. 패딩 분리 + 타일 단위 safe 체크 (분기 64회 → 1회)

### 개념
- **문제 1:** 패딩이 있을 때, 입력 인덱스 `ih = oh*stride_h - pad_h + kh`, `iw = ...` 가 경계 밖으로 나갈 수 있음. 원래는 **가장 안쪽 루프(kh, kw) 안에서** `if (ih < 0 || ih >= h_in || ...) continue` 로 막음 → 매 픽셀·매 (kh,kw)마다 분기.
- **문제 2:** "이 픽셀이 패딩 없이 안전한가?"를 **픽셀마다(dh,dw마다)** 물어보면, 타일당 최대 64번 분기.

**해결:**  
1. **안전 영역 정의:**  
   `safe_oh_min/max`, `safe_ow_min/max` 를 한 번 계산 (oh, ow가 이 범위 안이면, 해당 (oh,ow)에 대한 컨볼루션에서 ih, iw가 항상 [0, h_in), [0, w_in) 안에 있음).  
2. **타일 단위 판단:**  
   "이 타일 전체가 안전 영역 안인가?"를 **타일당 한 번만** 체크:  
   `tile_is_safe = (oh0 >= safe_oh_min && oh_end <= safe_oh_max && ow0 >= safe_ow_min && ow_end <= safe_ow_max)`.  
3. **경로 분리:**  
   - `tile_is_safe == 1` → 타일 내 **모든 (dh,dw)** 에 대해 **경계 체크 없이** 빠른 경로만 사용. (64번 분기 → 0번)  
   - `tile_is_safe == 0` → 타일 내 (dh,dw)마다 `in_safe` 체크 후, safe면 빠른 경로, 아니면 경계용 경로(if continue) 사용.

### 코드상 변경
- **한 번 계산 (타일 진입 시):**  
  `safe_oh_min = (pad_h + stride_h - 1) / stride_h`,  
  `safe_oh_max = (h_in - k_h + pad_h) / stride_h`,  
  `safe_ow_min`, `safe_ow_max` 동일.  
  `tile_is_safe = (oh0 >= safe_oh_min && oh_end <= safe_oh_max && ow0 >= safe_ow_min && ow_end <= safe_ow_max)`.
- **분기 구조:**  
  - `if (tile_is_safe)` 블록: `ic → b → dh → dw` 안에서 **if 없이** 빠른 경로만 실행 (x_row++, w_row++, contrib, acc_ptr[b] += contrib).  
  - `else` 블록: 같은 `ic → b → dh → dw` 이지만, **(dh,dw)마다** `in_safe` 계산 후 `if (in_safe)` 빠른 경로 / `else` 경계 경로(ih, iw 범위 체크) 실행.

즉, **"패딩이 필요한지 여부를 타일 단위로 한 번만 묻고, 안전한 타일은 64번 분기 없이 한 경로만 탄다"**가 코드 변경이다.

---

## 6. 누적 버퍼 인덱싱 — acc_ptr 로 다차원 인덱싱 축소

### 개념
- **문제:** `conv2d_acc_buf[dh][dw][b]` 접근 시, 컴파일러/CPU가 내부적으로 `base + (dh*(TILE_W*OC_BLOCK) + dw*OC_BLOCK + b)*sizeof(float)` 같은 **여러 곱셈**을 할 수 있음.
- **해결:** (dh, dw)가 정해졌을 때 **한 번만** "이 행의 시작 주소"를 계산하고, b는 그 포인터에서 **오프셋 b**만 더해 사용. 즉, `base + b` 한 번의 연산으로 접근.

### 코드상 변경
- **Before:**  
  `conv2d_acc_buf[dh][dw][b] += contrib;`  
  (다차원 배열 직접 접근.)
- **After:**  
  `float* acc_ptr = &conv2d_acc_buf[dh][dw][0];`  
  `acc_ptr[b] += contrib;`  
  (같은 (dh, dw)에 대해 base는 한 번만 계산, b에 대해서는 포인터+오프셋만 사용.)

즉, **"(dh,dw)당 한 번 base 포인터를 잡고, b 차원만 포인터 오프셋으로 처리했다"**가 코드 변경이다.

---

## 7. (kh,kw) 누적은 레지스터(contrib), 버퍼 반영은 1회

### 개념
- **문제:** 안쪽 루프에서 매번 `conv2d_acc_buf[...] += (*x_row++) * (*w_row++)` 처럼 **메모리 읽기·덧셈·쓰기**를 반복하면, 메모리 대역폭과 레이턴시가 병목이 됨 (특히 호스트에서).
- **해결:** (kh, kw)에 대한 합을 **지역 변수 contrib**에만 누적한 뒤, (dh, dw, b) 한 칸에 대해 **한 번만** `acc_ptr[b] += contrib` 로 메모리에 반영. contrib는 레지스터에 둘 가능성이 높음.

### 코드상 변경
- **빠른 경로:**  
  `float contrib = 0.0f;`  
  `for (kh) { for (kw) contrib += (*x_row++) * (*w_row++); }`  
  `acc_ptr[b] += contrib;`  
- **경계 경로:** 동일하게 `float contrib = 0.0f;` 로 (kh,kw) 합을 구한 뒤 `acc_ptr[b] += contrib`.

즉, **"가장 안쪽 루프에서는 메모리 누적 버퍼를 건드리지 않고, 레지스터에 해당하는 contrib에만 쌓고, 바깥에서 한 번만 버퍼에 더한다"**가 코드 변경이다.

---

## 8. 전체 루프 구조 요약 (현재 코드 기준)

```
for (ni)
  for (oh0 by tile_h)   // 타일링
    for (ow0 by tile_w)
      for (oc0 by oc_block)   // OC 블록
        초기화: acc_buf[dh][dw][b] = bias or 0
        tile_is_safe = (타일 전체가 safe 영역인가?)   // 분기 1회

        for (ic)
          for (b)   // 가중치 재사용: 같은 w_base로 64픽셀
            w_base = w + (oc0+b)*w_oc_stride + ic*w_ic_stride
            if (tile_is_safe)
              for (dh) for (dw)
                x_base, contrib = 0
                for (kh) for (kw) contrib += (*x_row++) * (*w_row++)  // strength reduction
                acc_ptr = &acc_buf[dh][dw][0]; acc_ptr[b] += contrib
            else
              for (dh) for (dw)
                in_safe = (이 픽셀만 safe?)
                contrib = (in_safe ? 빠른경로 : 경계경로)
                acc_ptr = &acc_buf[dh][dw][0]; acc_ptr[b] += contrib

        쓰기: y[...] = acc_buf[dh][dw][b]
```

---

## 9. 수정된 것만 한 줄로

| 항목 | 개념적 수정 | 코드적 수정 |
|------|-------------|-------------|
| 타일링 | 출력을 8×8 블록으로 나눠 캐시 재사용 | oh0/ow0 스텝, th/tw, oh=oh0+dh, ow=ow0+dw |
| OC 블록 | oc를 32개씩 묶어 타일×블록 단위 처리 | oc0 루프, n_oc, conv2d_acc_buf[][][], 마지막에 y에 복사 |
| 루프 순서 | 필터 1개를 64픽셀에 재사용 | ic → b → dh → dw (w_base는 ic,b마다 1회) |
| Strength reduction | 안쪽 루프에서 곱셈 제거 | x_row++, w_row++, stride 상수, x_base/w_base |
| 패딩/분기 | 타일이 전부 safe면 분기 0회 | tile_is_safe 1회, if(tile_is_safe) / else per-pixel in_safe |
| acc 인덱싱 | (dh,dw)당 base 1회, b는 오프셋 | acc_ptr = &acc_buf[dh][dw][0]; acc_ptr[b] += contrib |
| contrib | (kh,kw) 합은 레지스터, 버퍼는 1회 | float contrib; 루프 끝에 acc_ptr[b] += contrib |

이렇게 적용된 상태가 지금의 `conv2d.c`이다.
