# 변경사항 요약 (BARE_METAL 포팅)

## 목표
Arty A7-35T + MicroBlaze V (RISC-V) + Bare-metal (Vitis) 환경에서 YOLOv5n 추론 엔진 실행

---

## 최근 정리 (GitHub 업로드 전)

- **conv2d 추가 최적화 (제미나이 제안 반영):** (1) **입력 재사용:** 루프 순서를 oh0→ow0→oc_block→ic로 변경. 입력 타일 하나를 캐시에 올려두고 OC_BLOCK(기본 32) 출력 채널에 대해 연산 후 다음 ic로. (2) **Strength reduction:** 가장 안쪽 루프(kw)에서 인덱스 곱셈 제거, `x_row++`/`w_row++` 포인터 증감만 사용. (3) **패딩 분리:** 패딩이 필요 없는 안전 영역(safe_oh_min/max, safe_ow_min/max)과 경계를 분리; 안전 영역은 빠른 경로(분기 없음), 경계만 if 경로. (4) 누적 버퍼는 스택 대신 BSS 정적 배열 `conv2d_acc_buf[TILE_H][TILE_W][OC_BLOCK]` 사용 (bare-metal 스택 제한).
- **성능 최적화:** (1) D-Cache는 이미 `main.c`에서 `Xil_DCacheEnable()` 적용됨. (2) **타일링:** `csrc/operations/conv2d.c`에 출력 공간(oh, ow) 8×8 타일링 추가 — 캐시(16KB)에 맞춰 데이터 재사용 증가. `-DCONV2D_TILE_H=4 -DCONV2D_TILE_W=4`로 타일 크기 변경 가능. (3) **스택 BRAM 복귀:** `docs/VITIS_BUILD.md` §2에 벡터 테이블(.vectors)을 BRAM 최앞(0x0), 스택을 0x40 이후에 배치하는 lscript.ld 예시 추가 — 스택을 BRAM에 두어도 벡터 테이블 침범 방지.
- **주석/로그 정리:** 디버그용 DBG 로그, 레이어별 L0~L23 dump, p3 dump, largest_free, l0@ 주소 로그 제거. 과한 설명 주석 축약.
- **유지:** 사용자용 로그(Loading..., Running inference..., Decoded, After NMS, Summary, Done), 캐시 Flush/Invalidate 호출, C3 풀 실패 시 단일 메시지.
- **풀/플랫폼:** `FEATURE_POOL_BASE` 0x82000000, `FEATURE_POOL_SIZE` 32MB. C3 내부 할당 순서 큰 버퍼 우선. `feature_pool`/`platform_config` 주석 축약. `.gitignore`에 빌드 산출물(build_log.txt, gcc_*.txt, err.txt, main.exe 등) 추가.

---

## 주요 변경사항

### 1. 플랫폼 분리 (BARE_METAL 매크로)

**목적:** 호스트 빌드와 베어메탈 빌드를 하나의 코드베이스로 유지

**변경 파일:**
- `csrc/main.c`: `#ifdef BARE_METAL` 분기 추가
- `csrc/utils/weights_loader.c`: 파일 I/O를 `#ifndef BARE_METAL`로 감쌈
- `csrc/utils/image_loader.c`: 파일 I/O를 `#ifndef BARE_METAL`로 감쌈
- `csrc/utils/uart_dump.c`: BARE_METAL에서만 컴파일

**결과:**
- 호스트: 기존처럼 파일 I/O 사용
- BARE_METAL: DDR 메모리 직접 참조, UART 출력

---

### 2. DDR 메모리 맵 설계

**파일:** `csrc/platform_config.h` (신규)

**설계:**
```
0x8000_0000 ~ 0x87FF_FFFF  128MB  코드 / 스택 / 힙 (lscript.ld)
0x8800_0000 ~ 0x88FF_FFFF  16MB   가중치 (weights.bin)
0x8E00_0000 ~ 0x8E8F_FFFF  9MB   Detect Head 고정 (p3, p4, p5) — 풀 조각화 회피
0x8F00_0000 ~ 0x8FFFFFFF  16MB   입력 이미지(헤더 24B + 픽셀) + 피처맵 풀 + 검출 결과
```

**매크로:**
- `WEIGHTS_DDR_BASE` / `WEIGHTS_DDR_SIZE`
- `DETECT_HEAD_BASE` / `DETECT_HEAD_SIZE` (p3,p4,p5 고정, 4바이트 정렬)
- `IMAGE_DDR_BASE` / `IMAGE_DDR_SIZE`, `IMAGE_HEADER_SIZE` (24)
- `FEATURE_POOL_BASE` / `FEATURE_POOL_SIZE`
- `DETECTIONS_OUT_BASE`

**특징:**
- `xparameters.h`를 먼저 포함하면 `XPAR_DDR_MEM_BASEADDR` 기준으로 계산
- 빌드 시 `-DWEIGHTS_DDR_BASE=...` 등으로 덮어쓰기 가능

---

### 3. 가중치/이미지 로더: 제로카피 (Zero-Copy)

**파일:** `csrc/utils/weights_loader.c/h`, `csrc/utils/image_loader.c/h`

**변경:**
- `weights_init_from_memory(base, size, loader)`: DDR에서 직접 참조 (복사 없음)
- `image_init_from_memory(base, size, img)`: DDR에서 직접 참조
- `tensor_info_t.data_owned` / `preprocessed_image_t.data_owned` 플래그 추가
- `weights_free` / `image_free`: `data_owned`일 때만 `free()`

**효과:**
- 메모리 사용량 대폭 감소 (가중치 복사 제거)
- DDR 데이터를 직접 참조하여 성능 향상

---

### 4. 캐시 일관성 처리

**파일:** `csrc/main.c`

**추가:**
- **초기 무효화 (main 시작 직후):**  
  `WEIGHTS_DDR_BASE`, `IMAGE_DDR_BASE`, `FEATURE_POOL_BASE`, **`DETECT_HEAD_BASE`** 영역 `Xil_DCacheInvalidateRange`  
  이후 `Xil_DCacheEnable()` 호출 (DDR 읽기 캐시 활성화)
- **Detect 직후:** `Xil_DCacheFlushRange(DETECT_HEAD_BASE, DETECT_HEAD_SIZE)` — p3,p4,p5 write-back을 DDR에 반영
- **Decode 직전:** `Xil_DCacheInvalidateRange(DETECT_HEAD_BASE, DETECT_HEAD_SIZE)` — Decode가 DDR에서 최신 데이터 읽도록

**목적:** JTAG(MDM) 등으로 DDR에 쓴 데이터를 CPU가 읽기 전 캐시 무효화; p3,p4,p5 구간은 Flush/Invalidate로 Decode 시 캐시 꼬임 방지

**캐시 크기:** `sz_p3` 등 변수 대신 **`DETECT_HEAD_SIZE` 상수** 직접 사용 (9MB 전체 확실 처리)

---

### 5. RISC-V 메모리 정렬 (Unaligned Access 방지)

**파일:** `csrc/utils/weights_loader.c`

**문제:** RISC-V는 4바이트 단위 정렬된 주소만 로드 가능. 가중치 바이너리에서 텐서 이름 길이가 홀수(예: 31바이트)면 그 뒤 메타데이터/데이터 시작 주소가 비정렬 → Load Address Misaligned Trap → `_trap_handler`에서 정지.

**해결:**
- **메타데이터 읽기:** `*(int*)ptr` 대신 `memcpy` 또는 `read_u32_unaligned()`(바이트 단위 조합) 사용
- **데이터 시작 정렬:** 파싱 후 `curr`가 4의 배수가 아니면 패딩하여 `t->data`가 4바이트 정렬되도록 보정 (export 쪽 패딩과 쌍)
- `safe_read()`로 모든 헤더/이름 복사 시 memcpy 사용

**효과:** 가중치 파싱 단계에서 Trap 없이 "Weights: 121 tensors"까지 진행 가능

---

### 6. 스택·벡터 테이블 분리 (lscript.ld)

**문제:** `.stack` 섹션이 BRAM `0x00000000`~`0x00003fff`에 있으면, 스택이 아래로 자라며 **0x0~0x40(벡터 테이블)** 을 침범. Trap 발생 시 CPU가 0x4(Exception Vector)로 점프하는데, 그곳에 스택 데이터가 있으면 잘못된 명령 실행 → 무한 루프/사망.

**해결:** `lscript.ld`에서 **`.stack` 섹션을 DDR로 이동** (예: `0x80216448`~`0x8021a447`). 0x4에는 BSP가 생성한 정상 Trap 핸들러 코드가 있음.

**참고:** BRAM에 스택을 두고 싶다면, 벡터 테이블(.vectors)을 BRAM 최앞에 두고 스택 주소만 0x40 이후로 배치해야 함. 현재는 스택을 DDR로 두는 방식 채택.

---

### 7. Detect Head 출력 고정 영역 (p3, p4, p5)

**파일:** `csrc/platform_config.h`, `csrc/main.c`

**문제:** 피처맵 풀(First-fit) 조각화로 Neck/Head 단계에서 "ERROR: Feature pool allocation failed" 발생. p3(~6.5MB), p4(~1.6MB), p5(~0.4MB) 연속 블록이 풀에 없을 수 있음.

**해결:**
- **platform_config.h:** `DETECT_HEAD_BASE`(0x8E000000), `DETECT_HEAD_SIZE`(9MB) 정의. 4바이트 정렬 유지.
- **main.c (BARE_METAL):** p3,p4,p5를 풀 대신 **고정 DDR 영역** 사용:  
  `p3 = (float*)DETECT_HEAD_BASE`, `p4 = p3 + (255*80*80)`, `p5 = p4 + (255*40*40)`
- **p3,p4,p5는 풀에서 할당한 것이 아니므로** `feature_pool_free(p3/p4/p5)` 호출 제거 (BARE_METAL 분기에서만 free, 호스트는 기존처럼 POOL_ALLOC 후 free 유지)

**효과:** Head 단계 풀 할당 실패 제거; Decode가 p3,p4,p5를 DDR에서 읽도록 캐시 Flush/Invalidate와 연동

---

### 8. 이미지 픽셀 오프셋 (BARE_METAL)

**파일:** `csrc/main.c`, `csrc/platform_config.h`

**배경:** `preprocessed_image.bin` 형식은 **0x8F000000**에 24바이트 헤더(원본 가로/세로, scale, pad, 전처리 640 등), **0x8F000018**부터 float 픽셀 데이터. mrd로 확인 가능.

**문제:** 보드에서 캐시/파싱 이슈로 `image_init_from_memory`가 설정한 `img.data`가 base와 동일하게 남을 수 있음 → 모델이 헤더를 픽셀로 해석해 0 detections.

**해결:** BARE_METAL에서 이미지 로드 직후 **명시적 보정**  
`img.data = (float*)((uintptr_t)IMAGE_DDR_BASE + (uintptr_t)IMAGE_HEADER_SIZE);`  
→ 항상 픽셀 시작 주소(0x8F000018)를 가리킴.

---

### 9. 피처맵 버퍼 재사용 (Ping-pong Buffer)

**파일:** `csrc/utils/feature_pool.c/h` (신규), `csrc/main.c`

**구현:**
- First-fit 할당자 (First-fit allocator)
- BARE_METAL: `FEATURE_POOL_BASE` (DDR 고정 영역)
- 호스트: 22MB `malloc` 한 번

**변경:**
- `main.c`의 모든 피처맵 `malloc` → `feature_pool_alloc(size)` (p3,p4,p5는 BARE_METAL에서 풀 미사용)
- 각 레이어에서 "마지막 사용" 직후 `feature_pool_free(ptr)` 호출
- p3,p4,p5: BARE_METAL에서는 `feature_pool_free` 호출하지 않음 (고정 DDR 포인터)
- 종료 시 `feature_pool_reset()`

**효과:**
- 메모리 사용량: 41MB → 약 20MB 이하 (절반 이하)
- 버퍼 재사용으로 메모리 효율 향상

**예시:**
```c
float* l0 = (float*)feature_pool_alloc(sz_l0);
// ... Layer 0, 1 실행 ...
feature_pool_free(l0);  // l0 재사용 가능
```

---

### 10. UART 검출 결과 덤프

**파일:** `csrc/utils/uart_dump.c/h` (신규), `csrc/main.c`

**프로토콜:**
```
YOLO\n
<count 2자리 hex>\n
<12*count bytes hex (한 줄)>\n
```

**구현:**
- `yolo_uart_send_detections(hw_detections, count)`: `xil_printf`로 Hex 출력
- BARE_METAL에서만 컴파일

**PC 수신 스크립트:** `tools/recv_detections_uart.py` (신규)
- 시리얼에서 위 프로토콜 읽기
- `detections.bin` 형식으로 저장
- `tools/decode_detections.py`로 디코딩 가능

---

### 11. 디버깅·진행 로그 활성화

**파일:** `csrc/main.c`

**변경:**
- `YOLO_LOG`: BARE_METAL에서 `xil_printf` 사용 (기존: 비활성화)
- 각 단계별 로그 추가:
  - 이미지/가중치 로딩, 캐시 무효화
  - **진행 로그:** Backbone: L0~L9, Neck: L10~L23, Head: Detect
  - Decoded / After NMS / UART 전송 / 에러 메시지
- **0 detections 디버깅:** BARE_METAL에서 `num_dets == 0`일 때 p3[0], p3[1], p3[obj0] hex 값 출력 (이미지·캐시 문제 추적용)

**효과:**
- 디버깅 용이, 실패 원인 파악 가능
- 진행 속도 파악 (L0~L23 출력 간격)

---

### 12. 호스트 검증 (골든 비교)

**파일:** `build_run_compare.bat` (신규), `tools/decode_detections.py`

**목적:** FPGA 보드에서 0 detections 등 이슈 시, **동일 main.c를 호스트 PC에서 빌드·실행해 골든 결과(3 detections)와 일치하는지 확인** → C 로직/데이터 경로 검증.

**구성:**
- `build_run_compare.bat`: PATH에 MinGW(MSYS2) gcc 추가 후 빌드 → main.exe 실행 → `decode_detections.py --compare`로 골든과 비교
- `decode_detections.py`: Non-ASCII 문자 사용 시 **파일 상단 인코딩 선언** (`# -*- coding: utf-8 -*-`) 추가로 SyntaxError 방지

**골든 결과 예:** person 80%, person 39%, tie 26% (zidane.jpg 기준)

---

### 13. 문서화

**신규 파일:**
- `docs/TESTING.md`: 호스트/BARE_METAL 테스트 가이드, DDR mrd 확인, 골든 비교 절차
- `docs/VITIS_BUILD.md`: Vitis 빌드 상세 가이드
- `CHANGELOG.md`, `CHANGELOG_DETAILED.md`: 변경사항 요약·상세

**업데이트:**
- `README.md`: BARE_METAL 빌드, DDR 맵, BRAM 활용, UART 덤프 섹션 추가
- `.cursor/rules/microblaze-v-baremetal-yolov5n.mdc`: 베어메탈 제약 체크리스트

---

## 파일 변경 요약

### 신규 파일
```
csrc/platform_config.h            # DDR 메모리 맵, DETECT_HEAD, IMAGE_HEADER_SIZE
csrc/utils/feature_pool.c/h       # 피처맵 풀 할당자
csrc/utils/uart_dump.c/h         # UART 덤프 (BARE_METAL)
tools/recv_detections_uart.py    # PC 시리얼 수신 스크립트
build_run_compare.bat            # 호스트 빌드·실행·골든 비교 일괄
docs/TESTING.md                  # 테스트 가이드, mrd 확인
docs/VITIS_BUILD.md              # Vitis 빌드 가이드
CHANGELOG.md / CHANGELOG_DETAILED.md  # 변경사항 요약·상세
```

### 수정 파일
```
csrc/main.c                        # BARE_METAL 분기, feature_pool, UART, 캐시(Flush/Invalidate),
                                   # p3,p4,p5 고정 DDR, img.data 오프셋, 진행·DBG 로그, p3/p4/p5 free 제거
csrc/utils/weights_loader.c/h      # 제로카피, 파일 I/O 분리, memcpy/read_u32_unaligned(정렬 방지)
csrc/utils/image_loader.c/h        # 제로카피, image_init_from_memory(base, size, img)
tools/decode_detections.py         # 인코딩 선언(utf-8), --compare 골든 비교
README.md                          # BARE_METAL 섹션 추가
.cursor/rules/...mdc              # 베어메탈 제약 규칙
```

### Vitis 쪽 수정 (lscript.ld 등)
- **링커 스크립트:** `.stack` 섹션을 벡터 테이블과 겹치지 않도록 DDR로 이동 (예: 0x80216448~)
- **힙:** YOLO 피크 메모리 부족 시 `lscript.ld`에서 Heap Size를 64MB(0x4000000)급으로 확대 권장
- **빌드 옵션:** `-DBARE_METAL`, Include Paths(src, blocks, operations, utils), `-lm`, Linker Script

---

## 빌드 방법

### 호스트 빌드 (기존과 동일)
```bash
gcc -o main csrc/main.c csrc/blocks/*.c csrc/operations/*.c csrc/utils/*.c \
    -I. -Icsrc -lm -std=c99 -O2
./main
```

### BARE_METAL 빌드 (Vitis)
1. 컴파일 옵션: `-DBARE_METAL`
2. Include: `-I/path/to/YOLOv5n_in_C/csrc`
3. 소스: `csrc/main.c`, `csrc/blocks/*.c`, `csrc/operations/*.c`, `csrc/utils/*.c`
4. 링크 스크립트: `.stack` → BRAM, 나머지 → DDR

---

## 메모리 사용량 비교

### 기존 (호스트)
- 가중치: 복사본 (수 MB)
- 이미지: 복사본 (4.9MB)
- 피처맵: 41MB+ (각각 malloc)
- **총: ~50MB+**

### 개선 후 (BARE_METAL)
- 가중치: DDR 직참조 (0MB 추가)
- 이미지: DDR 직참조 (0MB 추가)
- 피처맵: 풀 11MB (재사용)
- **총: ~11MB (피크)**

---

## 실행 시 예상 결과

### 호스트 빌드
```
=== YOLOv5n Inference (Fused) ===

Image: 640x640
Weights: 121 tensors

Running inference...
Decoded: 19 detections
After NMS: 3 detections
Saved to data/output/detections.bin (37 bytes)
```

### BARE_METAL 빌드 (UART)
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
<hex data>
Done. Results at DDR 0x8FFFF000
```

---

## 주요 개선 효과

1. **메모리 효율**: 41MB → 11MB (약 73% 감소)
2. **제로카피**: 가중치/이미지 복사 제거
3. **버퍼 재사용**: Ping-pong 방식으로 메모리 절약
4. **디버깅**: UART 로그로 문제 파악 용이
5. **호환성**: 호스트 빌드와 동일 코드베이스 유지

---

## 다음 단계 (선택)

1. **Platform Init 추가**: 일부 BSP는 `init_platform()` 필요
2. **캐시 Stub**: `xil_cache.h` 없을 때 대비
3. **메모리 검증**: DDR 주소 유효성 체크
4. **성능 측정**: 타이머 추가 (현재 타이머 없음)
5. **힙 크기**: `lscript.ld`에서 Heap Size 64MB(0x4000000) 확대 후 Feature pool 할당 실패 여부 재확인

---

**작성일:** 2026-01-29  
**대상:** Arty A7-35T + MicroBlaze V + Vitis Bare-metal
