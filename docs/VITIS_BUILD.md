# Vitis BARE_METAL 빌드 가이드

## 현재 상태: 바로 빌드 가능하지만 런타임 전제조건 필요

코드는 BARE_METAL 매크로로 분기되어 있지만, **Vitis에서 바로 빌드하려면** 다음을 확인해야 합니다:

### 1. 컴파일 전제조건

**필수 BSP 헤더:**
- `xil_cache.h` - 캐시 무효화 (`Xil_DCacheInvalidateRange`)
- `xil_printf.h` - UART 출력 (`xil_printf`)
- `xparameters.h` - DDR 베이스 주소 (선택, 없으면 `platform_config.h` 기본값 사용)

**컴파일 옵션:**
```
-DBARE_METAL
-I/path/to/YOLOv5n_in_C/csrc
```

**최적화 레벨 (필수):**
- Vitis Application 프로젝트의 **Compiler Settings** 또는 **UserConfig.cmake**에서 **-O2** 또는 **-O3** 를 반드시 사용하세요.
- **-O0** (디버그용)으로 빌드하면 루프가 전혀 최적화되지 않아 **10배 이상** 느려집니다. 추론이 거의 멈춘 것처럼 보일 수 있습니다.

**소스 파일:**
- `csrc/main.c`
- `csrc/blocks/*.c`
- `csrc/operations/*.c`
- `csrc/utils/*.c` (모두 포함, `uart_dump.c`는 BARE_METAL에서만 컴파일됨)

### 2. 링크 스크립트 (lscript.ld) 및 MIG/Heap/Stack

**목표:** 코드·heap은 DDR **앞 32MB**(0x8000_0000~0x81FF_FFFF)만 쓰고, **0x8200_0000~0x83FF_FFFF(32MB)는 피처맵 풀 전용**으로 비워 둠. 스택은 BRAM에 두어 성능·안정 확보.

**권장 값 (괜찮음):**
| 항목 | 값 | 비고 |
|------|-----|------|
| **MIG(링커 DDR 영역) length** | **32MB** (0x2000000) | 링커가 .text/.data/.bss/heap을 이 범위 안에만 배치 → 0x82000000 이후는 풀 전용 |
| **Heap 크기** | **4MB** (0x400000) | malloc 등용. 피처맵은 풀(FEATURE_POOL_BASE) 사용하므로 4MB면 충분 |
| **Stack 크기** | **64KB** (0x10000) | BRAM에 배치 권장 (`.stack` → `local_memory_cntrl`) |

**필수 설정:**
- `.stack` → `local_memory_cntrl` (BRAM), 크기 64KB
- `.text`, `.data`, `.bss`, `.heap` → DDR (예: `mig_7series_0`), **MIG 영역 길이 32MB**
- Heap 크기: **4MB**

**lscript.ld에서 할 일:**
1. **MEMORY 블록** – DDR(MIG) 길이를 32MB로 제한:
```ld
MEMORY
{
  local_memory_cntrl : ORIGIN = 0x0, LENGTH = 0x10000   /* BRAM 64KB (스택용) */
  mig_7series_0     : ORIGIN = 0x80000000, LENGTH = 0x2000000   /* DDR 앞 32MB */
}
```
2. **스택** – BRAM, 64KB:
```ld
.stack : {
  . = ALIGN(8);
  _stack_end = .;
  . = . + 0x10000;   /* 64KB */
  _stack = .;
  *(.stack)
} > local_memory_cntrl
```
3. **Heap** – DDR, 4MB:
```ld
.heap : {
  . = ALIGN(8);
  _heap = .;
  . = . + 0x400000;   /* 4MB */
  _heap_end = .;
} > mig_7series_0
```

**Vitis에서 MIG length 지정:**  
플랫폼/보드 설정에서 MIG(DDR 컨트롤러) **주소 범위**를 0x80000000, **길이 32MB**(0x2000000)로 두면, 링커가 생성하는 lscript.ld의 `mig_7series_0` LENGTH도 32MB로 잡히는 경우가 많음. 수동으로 lscript.ld를 수정할 때는 위 MEMORY의 `LENGTH = 0x2000000`을 반드시 넣어 주면 됨.

#### 스택을 BRAM으로 복귀 (벡터 테이블과 분리)

**배경:** 과거에 `.stack`을 BRAM `0x00000000`~에 두었을 때, 스택이 아래로 자라며 **벡터 테이블(0x0~0x40)** 을 침범해 Trap 시 CPU가 0x4로 점프했을 때 스택 데이터를 실행하며 사망하는 문제가 있어 스택을 DDR로 옮긴 상태이다.

**해결:** 벡터 테이블을 BRAM 최앞에 고정하고, **스택 영역만 0x40 이후**에 배치하면 스택을 BRAM에 두어도 안전하다. (스택은 위로 자라지 않으므로 벡터 테이블을 덮지 않음.)

**lscript.ld 예시 (BRAM 64KB, 스택 복귀):**

1. **MEMORY** – BRAM과 DDR 구분 유지:
```ld
MEMORY
{
  local_memory_cntrl : ORIGIN = 0x0, LENGTH = 0x10000   /* BRAM 64KB */
  mig_7series_0     : ORIGIN = 0x80000000, LENGTH = 0x2000000   /* DDR 32MB */
}
```

2. **벡터 테이블** – BRAM 최앞(0x0), 최소 0x40 바이트 확보:
```ld
.vectors : {
  *(.vectors)
  . = ALIGN(0x40);   /* 0x0~0x3F 영역만 사용, 스택과 겹치지 않도록 */
  _vectors_end = .;
} > local_memory_cntrl
```

3. **스택** – BRAM 내 0x40 이후 ~ 64KB 끝까지 사용:
```ld
.stack : {
  . = ALIGN(8);
  _stack_end = .;    /* 스택 하한 (0x40 근처) */
  . = . + (0x10000 - 0x40);   /* 64KB - 벡터 영역 */
  _stack = .;        /* SP 초기값 = 스택 상한 */
  *(.stack)
} > local_memory_cntrl
```

**주의:** BSP에 따라 벡터 테이블 섹션 이름이 `.vectors`, `.vector_table` 등으로 다를 수 있음. 기존 lscript.ld에서 벡터용 섹션 이름을 확인한 뒤 위와 같이 BRAM 최앞에 두고, `.stack`은 반드시 그 뒤(0x40 이후)에 배치하면 된다.

### 3. 런타임 전제조건 (중요!)

**main.c 실행 전에 반드시:**

1. **DDR에 가중치 로드:**
   - `weights.bin` 내용을 `WEIGHTS_DDR_BASE` (기본: `0x88000000`)에 복사
   - 크기: `WEIGHTS_DDR_SIZE` (기본: 16MB) 이내

2. **DDR에 이미지 로드:**
   - `preprocessed_image.bin` 내용을 `IMAGE_DDR_BASE` (기본: `0x8F000000`)에 복사
   - 크기: `IMAGE_DDR_SIZE` (약 4.9MB)

3. **캐시:**
   - JTAG(MDM) 등으로 DDR에 쓴 후, CPU가 읽기 전에 캐시 무효화 필요 → `main.c` 초입에서 `Xil_DCacheInvalidateRange` 호출 (자동)
   - **Data Cache 활성화:** `main.c`에서 무효화 직후 `Xil_DCacheEnable()` 호출 (자동). 이게 없으면 매번 데이터 읽을 때마다 DDR까지 왕복해 **90% 이상 시간을 대기**에 씁니다.

### 4. main.c 실행 시 예상 결과

**성공 시:**
- 추론 완료 (타임아웃 없음)
- `DETECTIONS_OUT_BASE` (`0x8FFFF000`)에 결과 기록:
  - 1바이트: detection 개수 (0~255)
  - 이후: `hw_detection_t[]` (각 12바이트)
- UART로 Hex 덤프 전송:
  ```
  YOLO
  03
  <12*3 bytes hex>
  ```

**실패 시:**
- `image_init_from_memory` 실패 → `return 1` (조용히 종료, YOLO_LOG 비활성화)
- `weights_init_from_memory` 실패 → `return 1`
- `feature_pool_alloc` 실패 → `return 1`

**개선 완료:**
- ✅ `YOLO_LOG`가 `xil_printf`로 활성화됨 (에러 메시지 출력)
- ✅ 각 단계별 로그 출력 (로딩, 추론, 결과)

### 5. 성능 최적화 (캐시·타일링·스택 BRAM)

| 항목 | 상태 | 비고 |
|------|------|------|
| **D-Cache Enable** | ✅ 적용됨 | `main.c` 초입에서 `Xil_DCacheInvalidateRange` 후 `Xil_DCacheEnable()` 호출. 비활성화 시 DDR 왕복으로 대기 시간이 크게 늘어남. |
| **Write-Back** | BSP 기본 | D-Cache 활성화 시 Xilinx BSP는 보통 Write-Back 사용. 별도 설정 불필요. |
| **타일링 (Tiling)** | ✅ 적용됨 | `csrc/operations/conv2d.c`에서 출력 공간(oh, ow)을 8×8 타일로 나누어 연산. 캐시(예: 16KB)에 맞춰 데이터 재사용을 늘려 메모리 접근을 줄임. 타일 크기 변경: `-DCONV2D_TILE_H=4 -DCONV2D_TILE_W=4` 등. |
| **스택 BRAM** | 선택 | 스택을 BRAM에 두면 함수 호출·지역 변수 접근이 빨라짐. 위 §2 "스택을 BRAM으로 복귀" 참고. 벡터 테이블을 BRAM 최앞(0x0), 스택을 0x40 이후에 두어야 함. |

### 6. 추가 개선 (선택)

**B. Platform Init 추가 (선택):**

일부 BSP는 `init_platform()` 필요:
```c
#ifdef BARE_METAL
#include "xparameters.h"
#include "xil_cache.h"
void init_platform(void);  // BSP 제공
#endif

int main(...) {
#ifdef BARE_METAL
    init_platform();  // UART 초기화 등
#endif
    ...
}
```

**C. xil_cache.h 없을 때 Stub (선택):**

BSP에 캐시 헤더가 없으면:
```c
#ifdef BARE_METAL
#ifndef XIL_CACHE_H
#define Xil_DCacheInvalidateRange(addr, len) ((void)0)
#endif
#include "xil_cache.h"
#endif
```

### 7. 실제 테스트 순서

1. **Vitis 프로젝트 생성:**
   - MicroBlaze V 프로세서
   - DDR 컨트롤러 (MIG) 추가
   - UART 추가

2. **소스 추가:**
   - `csrc/` 전체 추가
   - 컴파일 옵션: `-DBARE_METAL`

3. **링크 스크립트 수정:**
   - MIG(DDR) length = 32MB, heap = 4MB, `.stack` = 64KB → BRAM (위 §2 참고)

4. **빌드:**
   - 컴파일/링크 성공 확인

5. **DDR 데이터 준비:**
   - JTAG/MDM으로 `weights.bin` → `0x88000000`
   - `preprocessed_image.bin` → `0x8F000000`

6. **실행:**
   - 디버거로 실행 또는 부팅
   - UART 모니터링 (115200 baud)
   - `DETECTIONS_OUT_BASE` 메모리 확인

### 8. 예상 출력 (UART)

**성공 시:**
```
=== YOLOv5n Inference (Fused) ===

Loading image from DDR 0x8F000000...
Image: 640x640
Loading weights from DDR 0x88000000 (size 16 MB)...
Weights: 121 tensors

Running inference...
Decoded: 19 detections
After NMS: 3 detections
Sending 3 detections to UART...
YOLO
03
01DA01E801DA01E8005000...
Done. Results at DDR 0x8FFFF000
```

**실패 시 (예: DDR에 데이터 없음):**
```
=== YOLOv5n Inference (Fused) ===

Loading image from DDR 0x8F000000...
ERROR: Failed to load image from DDR
```
(종료, return 1)

### 9. 디버깅 팁

**메모리 확인:**
- `WEIGHTS_DDR_BASE`에 가중치가 있는지 확인
- `IMAGE_DDR_BASE`에 이미지가 있는지 확인
- `FEATURE_POOL_BASE`가 유효한지 확인

**캐시 문제:**
- DDR에 쓴 직후 `Xil_DCacheInvalidateRange` 호출 확인
- 또는 캐시 비활성화

**UART 문제:**
- BSP에서 UART 초기화 확인
- `xil_printf`가 실제로 출력되는지 테스트

### 10. 결과를 detections.txt로 변환

추론이 끝나면 **DDR `DETECTIONS_OUT_BASE`** 또는 **UART**로 검출 결과(1바이트 개수 + 12×N 바이트)가 나옵니다. 이를 `detections.txt`(및 시각화 `detections.jpg`)로 만들려면:

#### 방법 A: UART로 받은 경우 (권장)

1. **한 번에 (UART → detections.txt + detections.jpg):**
   ```bash
   python tools/uart_to_detections_txt.py --port COM3
   ```
   - `data/output/detections.bin`, `detections.txt`, `detections.jpg` 생성
   - 시각화 생략: `--no-viz` 추가

2. **두 단계로:**
   ```bash
   python tools/recv_detections_uart.py --port COM3 --out data/output/detections.bin
   python tools/decode_detections.py --c-bin data/output/detections.bin --out-dir data/output
   ```

#### 방법 B: DDR에서 덤프한 경우

1. XSCT(JTAG) 등으로 **`DETECTIONS_OUT_BASE`**(기본 `0x8FFFF000`)에서 **1 + 12×count** 바이트를 읽어 `detections.bin`으로 저장 (첫 1바이트가 개수).
2. **바이너리 → txt 변환:**
   ```bash
   python tools/decode_detections.py --c-bin data/output/detections.bin --out-dir data/output
   ```

#### detections.txt 형식

- 한 줄에 한 검출: `class_id class_name confidence x y w h`
- x, y, w, h는 640×640 입력 기준 픽셀 (중심 좌표 및 너비/높이)

---

**결론:** 코드는 BARE_METAL 구조로 되어 있지만, **런타임 전제조건(DDR 데이터 준비)이 충족되어야** 정상 작동합니다. 바로 빌드는 가능하지만, 실행 전 DDR 준비가 필수입니다.
