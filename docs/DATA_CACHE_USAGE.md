# Data Cache (D-Cache) 사용 정리

BARE_METAL 빌드에서 DDR 메모리와 캐시 일관성을 위해 사용하는 **Data Cache** 관련 API·구간·코드를 정리한 문서이다.

---

## 1. 사용 API (xil_cache.h)

| API | 역할 |
|-----|------|
| `Xil_DCacheInvalidateRange(addr, len)` | 지정 주소 범위의 **캐시 라인을 무효화**. 이후 CPU가 해당 주소를 읽으면 **DDR에서 최신 데이터**를 가져온다. (JTAG/MDM 등 외부에서 DDR에 쓴 직후 사용) |
| `Xil_DCacheFlushRange(addr, len)` | 지정 주소 범위의 **캐시에 있는 수정 데이터를 DDR에 반영(Write-Back)** 한 뒤, 필요 시 해당 캐시 라인을 무효화할 수 있다. CPU가 쓴 데이터를 DDR에 남기고 싶을 때 사용. |
| `Xil_DCacheEnable()` | **D-Cache 활성화**. 비활성화 시 매 읽기/쓰기가 DDR로 직접 가서 대기 시간이 크게 늘어남. |

- **헤더**: `#include "xil_cache.h"` (main.c, BARE_METAL 분기 내)
- **플랫폼**: Vitis BSP 제공. BSP에 없으면 stub으로 `Xil_DCacheInvalidateRange(addr,len)` → `(void)0` 등으로 무효화 가능.

---

## 2. DDR 메모리 영역 (캐시 대상)

`platform_config.h` 기준 주소·크기:

| 심볼 | 기본 주소 | 크기 | 용도 |
|------|-----------|------|------|
| `WEIGHTS_DDR_BASE` | 0x88000000 | 16MB | 가중치 (weights.bin) |
| `IMAGE_DDR_BASE` | 0x8F000000 | IMAGE_DDR_SIZE | 전처리 이미지 (헤더 24B + 3×640×640 float) |
| `FEATURE_POOL_BASE` | 0x82000000 | 32MB | 피처맵 풀 (l0~l23 등 중간 텐서) |
| `DETECT_HEAD_BASE` | 0x8E000000 | 9MB | Detect Head 출력 (p3, p4, p5) |
| `DETECTIONS_OUT_BASE` | 0x8FFFF000 근처 | 4KB 이내 | 검출 결과 (개수 + hw_detection_t[]) |

---

## 3. 구간별 D-Cache 사용

### 3.1 main() 진입 직후 — Invalidate + Enable

**위치**: `main.c` 내, `#ifdef BARE_METAL` 블록, 이미지/가중치 로드 **직전**.

**목적**  
JTAG(MDM) 등으로 DDR에 미리 써 둔 **가중치·이미지·피처맵 풀·Detect 영역**을 CPU가 읽기 전에 캐시와 일치시키기 위해 해당 영역을 **무효화**하고, 이후 모든 DDR 접근에 **D-Cache를 사용**하도록 **Enable** 한다.

**코드** (`csrc/main.c`):

```c
#ifdef BARE_METAL
    Xil_DCacheInvalidateRange((uintptr_t)WEIGHTS_DDR_BASE, (unsigned int)WEIGHTS_DDR_SIZE);
    Xil_DCacheInvalidateRange((uintptr_t)IMAGE_DDR_BASE, (unsigned int)IMAGE_DDR_SIZE);
    Xil_DCacheInvalidateRange((uintptr_t)FEATURE_POOL_BASE, (unsigned int)FEATURE_POOL_SIZE);
    Xil_DCacheInvalidateRange((uintptr_t)DETECT_HEAD_BASE, (unsigned int)DETECT_HEAD_SIZE);
    Xil_DCacheEnable();

    YOLO_LOG("Loading image from DDR 0x%08X...\n", (unsigned int)IMAGE_DDR_BASE);
    // ...
```

| 호출 | 주소 | 크기 |
|------|------|------|
| Invalidate | WEIGHTS_DDR_BASE | WEIGHTS_DDR_SIZE (16MB) |
| Invalidate | IMAGE_DDR_BASE | IMAGE_DDR_SIZE |
| Invalidate | FEATURE_POOL_BASE | FEATURE_POOL_SIZE (32MB) |
| Invalidate | DETECT_HEAD_BASE | DETECT_HEAD_SIZE (9MB) |
| Enable | — | — |

---

### 3.2 추론 시작 직전 — 이미지·가중치 재무효화 (선택)

**위치**: `main.c`, `POOL_ALLOC` 등 추론 루프 **직전**, `#ifdef BARE_METAL` 블록.

**목적**  
호스트에서 같은 바이너리로 여러 번 돌리거나, 디버깅 중 DDR을 다시 채웠을 수 있을 때, **이미지·가중치**만 다시 무효화해 CPU가 최신 DDR 내용을 보도록 한다. (필요 없으면 제거 가능)

**코드** (`csrc/main.c`):

```c
#ifdef BARE_METAL
    Xil_DCacheInvalidateRange((uintptr_t)IMAGE_DDR_BASE, (unsigned int)IMAGE_DDR_SIZE);
    Xil_DCacheInvalidateRange((uintptr_t)WEIGHTS_DDR_BASE, (unsigned int)WEIGHTS_DDR_SIZE);
    {
        // DBG 로그 등
    }
#endif
    YOLO_LOG("Running inference...\n");
```

| 호출 | 주소 | 크기 |
|------|------|------|
| Invalidate | IMAGE_DDR_BASE | IMAGE_DDR_SIZE |
| Invalidate | WEIGHTS_DDR_BASE | WEIGHTS_DDR_SIZE |

---

### 3.3 레이어별 출력 직후 — Flush (L0~L23)

**위치**: `main.c` 내, 각 레이어(L0~L23) 연산 **직후**, `#ifdef BARE_METAL` 블록.

**목적**  
해당 레이어 출력 버퍼(예: `l0`, `l1`, …)가 **피처맵 풀(DDR)** 에 있을 때, CPU가 캐시에 써 둔 내용을 **DDR에 반영**해 두기 위해 **Flush** 한다. (다음 레이어나 풀 해제 후 재사용 시 DDR에서 읽을 수 있도록)

**코드** (`csrc/main.c`): 레이어마다 다음 패턴 반복.

```c
    conv_block_nchw_f32(...);
    layer_cycles[0] = timer_delta64(t_layer, timer_read64());
    LAYER_LOG(0, layer_cycles[0], &l0[0]);
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)l0, 16);
#endif
```

- **참고**: 현재는 버퍼 전체가 아니라 **16바이트**만 Flush (`(uintptr_t)l0, 16`). 최소한 해당 캐시 라인을 DDR에 쓰는 용도. 버퍼 전체를 반영하려면 해당 레이어 출력 크기(예: `sz_l0`)로 Flush 하도록 변경 가능.

| 구간 | 포인터 | Flush 크기 (현재) |
|------|--------|-------------------|
| L0 직후 | l0 | 16 |
| L1 직후 | l1 | 16 |
| … | … | 16 |
| L23 직후 | l23 | 16 |

---

### 3.4 Detect 직후 — Flush (p3, p4, p5)

**위치**: `main.c`, Detect Head 연산 **직후**, Decode **직전**, `#ifdef BARE_METAL` 블록.

**목적**  
Detect Head 출력 **p3, p4, p5**가 `DETECT_HEAD_BASE` 구간(DDR)에 있을 때, CPU가 캐시에 써 둔 p3/p4/p5를 **DDR에 반영**한다. 이어서 **Decode**가 p3/p4/p5를 읽을 때 DDR(또는 무효화 후 캐시)에서 일관된 데이터를 보도록 한다.

**코드** (`csrc/main.c`):

```c
    YOLO_LOG("Detect\n");
    cycles_head = timer_delta64(t_stage_start, timer_read64());
#ifdef BARE_METAL
    YOLO_LOG("  det %llu ms\n", LAYER_MS_INT(cycles_head));
#else
    // ...
#endif
#ifdef BARE_METAL
    Xil_DCacheFlushRange((uintptr_t)DETECT_HEAD_BASE, (unsigned int)DETECT_HEAD_SIZE);
    __sync_synchronize();
#endif
    feature_pool_free(l17);
    // ...
```

| 호출 | 주소 | 크기 |
|------|------|------|
| Flush | DETECT_HEAD_BASE | DETECT_HEAD_SIZE (9MB) |

- **Decode 직전 Invalidate**: 현재 코드에는 **Decode 직전** `Xil_DCacheInvalidateRange(DETECT_HEAD_BASE, DETECT_HEAD_SIZE)` 호출은 없다. Flush만으로도 캐시→DDR 반영이 되고, Decode가 같은 캐시를 읽으면 동일 데이터이므로, BSP 동작에 따라 생략 가능. Decode가 항상 DDR 기준으로 읽어야 한다면 Flush 직후에 Invalidate를 추가할 수 있다.

---

### 3.5 결과 기록 후 — D-Cache 재활성화

**위치**: `main.c`, `DETECTIONS_OUT_BASE`에 검출 결과 기록 및 UART 전송 **직후**, `#ifdef BARE_METAL` 블록.

**목적**  
일부 BSP/드라이버가 UART 등 작업 중 D-Cache를 비활성화했을 수 있으므로, 이후 코드나 재실행을 위해 **D-Cache를 다시 켠다**.

**코드** (`csrc/main.c`):

```c
        memcpy(out, &hw, sizeof(hw_detection_t));
        out += sizeof(hw_detection_t);
        }
        YOLO_LOG("Sending %d detections to UART...\n", (int)count);
        yolo_uart_send_detections(...);
        YOLO_LOG("Done. Results at DDR 0x%08X\n", (unsigned int)DETECTIONS_OUT_BASE);
        Xil_DCacheEnable();
```

| 호출 | 설명 |
|------|------|
| Xil_DCacheEnable() | D-Cache 재활성화 |

---

## 4. 요약 표

| 구간 | 시점 | API | 대상 주소 | 크기 |
|------|------|-----|-----------|------|
| main 진입 | 이미지/가중치 로드 직전 | Invalidate | WEIGHTS_DDR_BASE | WEIGHTS_DDR_SIZE |
| | | Invalidate | IMAGE_DDR_BASE | IMAGE_DDR_SIZE |
| | | Invalidate | FEATURE_POOL_BASE | FEATURE_POOL_SIZE |
| | | Invalidate | DETECT_HEAD_BASE | DETECT_HEAD_SIZE |
| | | Enable | — | — |
| 추론 직전 | Running inference... 직전 | Invalidate | IMAGE_DDR_BASE | IMAGE_DDR_SIZE |
| | | Invalidate | WEIGHTS_DDR_BASE | WEIGHTS_DDR_SIZE |
| 레이어 L0~L23 | 각 레이어 연산 직후 | Flush | l0~l23 | 16 (각) |
| Detect 직후 | Decode 직전 | Flush | DETECT_HEAD_BASE | DETECT_HEAD_SIZE |
| 결과 기록 후 | UART 전송 직후 | Enable | — | — |

---

## 5. 관련 파일

- **캐시 호출**: `csrc/main.c` (BARE_METAL 분기 내)
- **주소/크기 정의**: `csrc/platform_config.h`
- **BSP 헤더**: `xil_cache.h` (Vitis BSP)
- **빌드/캐시 정책**: [VITIS_BUILD.md](VITIS_BUILD.md) §3 런타임 전제조건, §5 성능 최적화
