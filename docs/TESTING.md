# 테스트 가이드

## 호스트 빌드 테스트 (BARE_METAL 없이)

### 1. 전체 추론 테스트

```bash
# 빌드 (BARE_METAL 없이)
gcc -o main csrc/main.c csrc/blocks/*.c csrc/operations/*.c csrc/utils/*.c \
    -I. -Icsrc -lm -std=c99 -O2

# 실행 (파일 I/O 경로 사용)
./main
# 출력: data/output/detections.bin

# 결과 확인
python tools/decode_detections.py
```

**체크리스트:**
- [ ] 컴파일 성공 (feature_pool.c 포함)
- [ ] 실행 성공 (feature_pool_init에서 22MB malloc)
- [ ] `data/output/detections.bin` 생성
- [ ] 검출 결과가 Python 참조와 일치

### 2. 단위 테스트 (기존)

기존 테스트들은 `weights_load_from_file`을 사용하므로 **변경 없이** 작동합니다.

```bash
# 예: Conv 블록 테스트
gcc -o tests/test_conv tests/test_conv.c \
    csrc/blocks/conv.c csrc/operations/conv2d.c csrc/operations/silu.c \
    csrc/utils/weights_loader.c \
    -I. -Icsrc -lm -std=c99 -O2
./tests/test_conv
```

**체크리스트:**
- [ ] `test_conv` 통과
- [ ] `test_c3` 통과
- [ ] `test_sppf` 통과
- [ ] `test_detect` 통과
- [ ] `test_decode` 통과
- [ ] `test_nms` 통과
- [ ] `test_upsample` 통과

### 3. Feature Pool 동작 확인

호스트 빌드에서 `feature_pool`은:
- `feature_pool_init()`: 22MB `malloc` 한 번
- `feature_pool_alloc(size)`: First-fit 할당
- `feature_pool_free(ptr)`: 반환 (재사용 가능)
- `feature_pool_reset()`: 전체 해제

**메모리 사용량:**
- 기존: 41MB+ (각 피처맵 malloc)
- 현재: 22MB 풀에서 재사용 → **절반 이하**

## BARE_METAL 빌드 테스트 (Vitis)

### 1. 컴파일 체크

```bash
# Vitis 애플리케이션 프로젝트에서
# 컴파일 옵션: -DBARE_METAL
# 소스: csrc/main.c, csrc/blocks/*.c, csrc/operations/*.c, csrc/utils/*.c
```

**체크리스트:**
- [ ] `-DBARE_METAL` 정의 시 컴파일 성공
- [ ] `xil_cache.h` 포함 가능 (캐시 무효화)
- [ ] `xil_printf.h` 포함 가능 (UART 덤프)
- [ ] `platform_config.h` DDR 주소 정의 확인

### 2. 링크 체크

- [ ] `.stack` 섹션이 `local_memory_cntrl`(BRAM)로 배치
- [ ] `.text`, `.data`, `.bss`, heap이 DDR로 배치
- [ ] Heap 크기 충분 (피처맵 풀 11MB + 여유)

### 3. 런타임 체크

**DDR 메모리 준비:**
- [ ] `weights.bin`을 `WEIGHTS_DDR_BASE`에 로드
- [ ] `preprocessed_image.bin`을 `IMAGE_DDR_BASE`에 로드

**실행 후:**
- [ ] 추론 완료 (타임아웃 없음)
- [ ] `DETECTIONS_OUT_BASE`에 결과 기록 (1 byte count + hw_detection_t[])
- [ ] UART로 Hex 덤프 전송 (`YOLO\n` → count → hex)

**UART 수신 테스트:**
```bash
# PC에서 시리얼 수신
python tools/recv_detections_uart.py --port COM3 --out data/output/detections_uart.bin
python tools/decode_detections.py data/output/detections_uart.bin
```

### 4. DDR 적재 확인 (xsdb)

보드에서 0 detections가 나올 때, **이미지/가중치가 올바른 주소에 올라갔는지** xsdb로 확인할 수 있다.

**이미지 (0x8F000000):**
- 형식: 헤더 24바이트 (original_w, original_h, scale, pad_x, pad_y, size) + `3*640*640*4` 바이트 float
- 선두 24바이트: `mrd 0x8F000000 6` → 처음 4바이트가 원본 너비(예: 0x000001E0=480), 다음이 높이 등
- 이미지 데이터 첫 float 몇 개: `mrd 0x8F000018 8` → 0이 아닌 값들이 보이면 적재된 것

**가중치 (0x88000000):**
- 형식: [name_len(4B), name..., align, shape..., float data...] 반복
- `mrd 0x88000000 4` → 첫 텐서 name_len (작은 정수, 예: 0x0000001A)
- 그 다음 바이트들이 텐서 이름(ASCII)이면 weights.bin이 올바르게 올라간 것
- **RISC-V 정렬:** weights.bin은 반드시 `tools/export_weights_to_bin.py`로 생성 (각 텐서 float 데이터 직전 4바이트 패딩). 비정렬 시 가중치가 0으로 읽혀 결과가 Bias(-4.375)만 남음.
- **중간/후반 데이터:** `mrd 0x88800000 16` 등으로 중간 지점 확인. FFF.../EFF... 같은 쓰레기면 dow -data 미완료·손상 의심 → 재적재.

**Detect 출력 p3 (0x8E000000):**
- 추론 완료 후 `mrd 0x8E000000 16` → 전부 0이면 Detect가 쓴 데이터가 DDR에 안 내려왔거나 Decode가 캐시만 읽은 상황. 0이 아닌 float 비트(예: 3DCF0000, 3E5ED8FC)가 보이면 DDR에는 올바른 값이 있음 → 이때도 0 detections면 **Decode가 캐시(옛 데이터)를 읽은 것**일 가능성이 큼.

## 호스트에서 golden 결과 확인

C 파이프라인이 정상인지 확인하려면 **BARE_METAL 없이** 호스트에서 실행해 보라. golden과 같은 3건(person×2, tie×1)이 나와야 한다.

```bash
# 프로젝트 루트에서
gcc -o main csrc/main.c csrc/blocks/*.c csrc/operations/*.c csrc/utils/*.c \
    -I. -Icsrc -lm -std=c99 -O2
./main
python tools/decode_detections.py data/output/detections.bin
# data/output/detections.txt 에 3건 나오면 OK (golden: person 0.8, person 0.388, tie 0.267)
```

- 호스트에서 3건 나오고 보드에서 0건이면: **보드 쪽만** 문제 (DDR 적재 또는 캐시 일관성).
- 호스트에서도 0건이면: 이미지/가중치 파일 경로·형식·알고리즘 점검.

## 알려진 이슈

### 호스트 빌드
- `feature_pool`이 22MB를 한 번에 할당하므로 메모리가 부족한 환경에서는 실패할 수 있음
- 해결: `feature_pool.c`의 `pool_size`를 줄이거나, 환경에 맞게 조정

### BARE_METAL 빌드
- `xil_cache.h` / `xil_printf.h`가 없는 BSP에서는 컴파일 실패
- 해결: 해당 BSP에 맞게 헤더 경로/이름 조정 또는 stub 구현
- **메모리 경계:** Arty A7-35T DDR3 = 256MB (0x8000_0000 ~ 0x8FFF_FFFF). `FEATURE_POOL_BASE + FEATURE_POOL_SIZE`가 0x8FFF_F000(결과 영역)을 넘으면 안 됨. 0x8F60_0000 + 11MB = 0x9010_0000 → **초과**이므로 풀은 0x8F50_0000(이미지 5MB 뒤)부터 사용.
- **가중치 정렬:** `weights_init_from_memory` 실패(비정렬 t->data) 시 weights.bin을 `export_weights_to_bin.py`로 다시 생성.

## 빠른 체크

**호스트 빌드가 작동하는지:**
```bash
# 최소 테스트: 컴파일만
gcc -c csrc/utils/feature_pool.c -I. -Icsrc -std=c99
# 성공하면 OK
```

**BARE_METAL 경로가 제대로 분리되었는지:**
```bash
# BARE_METAL 없이 컴파일 시 uart_dump.c는 빈 파일 (컴파일 OK)
gcc -c csrc/utils/uart_dump.c -I. -Icsrc -std=c99
# 성공하면 OK
```
