# YOLOv5n Pure C Implementation

순수 C로 구현한 YOLOv5n(nano) 객체 탐지 추론 엔진. 외부 라이브러리 없이 동작하며, 호스트 빌드와 **Bare-metal(FPGA)** 빌드를 하나의 코드베이스로 지원한다.

## 목표

- **최종**: MicroBlaze V(RISC-V) 등 FPGA에서 YOLOv5n 추론 실행
- **제약**: OpenCV/OpenBLAS 등 미사용, 순수 C만 사용
- **상태**: Python YOLOv5n과 동일한 추론 결과 (호스트·보드 검증 완료)
- **단계별 시간 측정**: 레이어(L0~L23)·backbone/neck/head·decode·nms·total을 ms 단위로 출력. 호스트는 마이크로초 기반, 보드는 mcycle 기반으로 동일 구간 비교 가능.

## 폴더 구조

```
YOLOv5n_in_C/
├── assets/                     # 모델 파일
│   ├── yolov5n.pt              # PyTorch 원본 모델
│   └── weights.bin             # C용 변환된 가중치 (Fused)
│
├── csrc/                        # C 소스 코드
│   ├── main.c                  # 메인 추론 파이프라인
│   ├── platform_config.h       # BARE_METAL DDR 맵 / 매크로
│   │
│   ├── blocks/                  # 고수준 블록
│   │   ├── conv.c/h            # Conv 블록 (Conv2D + Bias + SiLU)
│   │   ├── c3.c/h              # C3 블록 (cv1 + cv2 + Bottleneck + cv3)
│   │   ├── sppf.c/h            # SPPF 블록 (Spatial Pyramid Pooling Fast)
│   │   ├── detect.c/h          # Detect Head (1×1 Conv × 3 스케일)
│   │   ├── decode.c/h          # Anchor-based Decode + hw_detection_t 정의
│   │   └── nms.c/h             # Non-Maximum Suppression
│   │
│   ├── operations/              # 저수준 연산
│   │   ├── conv2d.c/h          # 2D Convolution (타일링·가중치 재사용·strength reduction 등 최적화)
│   │   ├── silu.c/h            # SiLU 활성화 함수
│   │   ├── bottleneck.c/h      # Bottleneck 모듈
│   │   ├── concat.c/h          # 채널 방향 Concat
│   │   ├── maxpool2d.c/h       # 2D Max Pooling
│   │   └── upsample.c/h        # Nearest Neighbor 2× Upsampling
│   │
│   └── utils/                   # 유틸리티
│       ├── weights_loader.c/h  # weights.bin 로더 (DDR 제로카피 지원)
│       ├── image_loader.c/h    # 전처리된 이미지 로더 (DDR 제로카피 지원)
│       ├── feature_pool.c/h    # 피처맵 풀 할당자 (버퍼 재사용)
│       ├── mcycle.h            # 단계별 시간/사이클 측정 (mcycle 호스트 타이머)
│       └── uart_dump.c/h       # UART 검출 결과 덤프 (BARE_METAL)
│
├── data/
│   ├── image/                   # 입력 이미지
│   │   └── zidane.jpg
│   ├── input/                   # 전처리된 입력
│   │   └── preprocessed_image.bin
│   └── output/                  # 추론 결과
│       ├── detections.bin      # C 결과 (HW 바이너리 포맷)
│       ├── detections.txt      # C 결과 (텍스트)
│       ├── detections.jpg      # C 결과 시각화
│       └── ref/                 # Python 참조 결과
│           ├── detections.bin
│           ├── detections.txt
│           └── detections.jpg
│
├── tools/                        # Python 도구
│   ├── export_weights_to_bin.py # PyTorch → weights.bin 변환
│   ├── preprocess_image_to_bin.py # 이미지 전처리
│   ├── run_python_yolov5n_fused.py # Python 참조 출력 생성
│   ├── decode_detections.py     # bin → txt 변환 + 시각화
│   ├── recv_detections_uart.py  # UART 수신 → detections.bin (BARE_METAL용)
│   ├── uart_to_detections_txt.py # UART 수신 → detections.txt(.jpg) 한 번에
│   ├── verify_weights_bin.py    # weights.bin 형식 검증
│   ├── reweight_align4.py       # weights.bin 4바이트 정렬 패딩 추가
│   └── gen_test_vectors.py      # 테스트 벡터 생성
│
├── tests/                        # 단위 테스트
│   ├── test_*.c                 # 각 블록별 테스트
│   └── test_vectors_*.h         # 테스트 벡터
│
├── docs/                         # 문서 (상세 가이드)
│   ├── DATA_CACHE_USAGE.md      # D-Cache 사용 방법·구간별 코드
│   ├── VITIS_BUILD.md           # Vitis Bare-metal 빌드·메모리 맵·링커 스크립트
│   ├── TESTING.md               # 테스트 방법
│   └── CONV2D_OPTIMIZATION.md   # conv2d 최적화 상세 (개념·코드)
├── CHANGELOG.md                  # 변경 이력
└── README.md                     # 본 문서
```

## 빌드 및 실행

### 호스트 (Linux/macOS/Windows)

**1. 준비**  
- **Python 도구용 가상환경** (numpy, torch, Pillow 등): 프로젝트 루트에서  
  `python3 -m venv .venv` 후 `.venv/bin/pip install -r requirements.txt`  
  (Windows: `.venv\Scripts\pip install -r requirements.txt`)  
  도구 실행 시 `.venv/bin/python tools/...` 사용 권장.
- 전처리 이미지: `tools/preprocess_image_to_bin.py` → `data/input/preprocessed_image.bin`  
- 가중치: `tools/export_weights_to_bin.py` → `assets/weights.bin`

**2. 빌드**

```bash
gcc -o main csrc/main.c csrc/blocks/*.c csrc/operations/*.c csrc/utils/*.c \
    -I. -Icsrc -lm -std=c99 -O2
```

W8A32(가중치 INT8) 사용 시: `tools/quantize_weights.py`로 `weights_w8.bin` 생성 후 (scale은 w8 내부 포함)

**FP32 vs W8A32 호스트 비교**: `./run_compare_host.sh` 실행 시 FP32(수정 전) → W8A32(수정 후) 순으로 빌드·실행 후 `data/output/ref_fp32_detections.bin`·`ref_fp32_log.txt`와 `detections.bin`·`w8_log.txt`를 저장하고, `tools/compare_fp32_w8.py`로 검출 개수·항목별 비교 및 L0/total 로그를 출력한다.  
`-DUSE_WEIGHTS_W8` 추가하여 빌드. (예: `-O2 -DUSE_WEIGHTS_W8`)

Windows(예: MinGW)에서는 `build_host.bat` 또는 위와 동일한 gcc 명령으로 빌드.

**3. 실행**

프로젝트 루트에서:

```bash
./main
```

Windows: `main.exe`

**4. 결과**  
- 입력: `data/input/preprocessed_image.bin`, 가중치: `assets/weights.bin` (파일에서 로드)  
- 출력: `data/output/detections.bin` (1바이트 개수 + 12바이트×N 검출)  
- 콘솔에 **각 레이어/연산을 지날 때마다** `  L0 123.45 ms (0x...)` 형태로 즉시 출력되며, 마지막에 `[time] backbone=... ms ... total=... ms` 요약이 출력됨. BARE_METAL 보드의 동일 단위(ms) 출력과 직접 비교 가능.

### Bare-metal (Vitis, Arty A7 + MicroBlaze V 등)

- 컴파일 옵션: `-DBARE_METAL`, include: `csrc`
- 입력/가중치: DDR 고정 주소에서 직접 참조 (파일 I/O 없음)
- 출력: DDR `DETECTIONS_OUT_BASE` 버퍼 + UART Hex 덤프
- CPU 클럭: `platform_config.h` 의 `CPU_MHZ` (기본 100MHz). **각 레이어/연산을 지날 때마다** `  L0 12345 ms (0x...)` 형태(정수 ms)로 즉시 출력되며, 마지막에 `[mcycle]`·`[time @ 100MHz]` 요약이 출력됨. xil_printf는 `%f` 미지원이라 보드에서는 정수 ms만 사용.

상세 메모리 맵, 캐시, 링커 스크립트, UART 프로토콜은 **[docs/VITIS_BUILD.md](docs/VITIS_BUILD.md)** 참고. **D-Cache 사용 방법·구간별 코드**는 **[docs/DATA_CACHE_USAGE.md](docs/DATA_CACHE_USAGE.md)** 참고.

## 단계별 시간 측정

추론 구간별로 **어디서 시간이 쓰이는지** 보기 위해, 레이어(L0~L23)·detect·decode·nms 단위로 시간을 측정한다. 호스트와 보드 모두 **동일한 구간**을 **ms** 단위로 출력해 비교할 수 있다.

### 측정 대상

- **레이어 L0~L23**: 각 레이어의 **연산만** 측정 (POOL_ALLOC/`feature_pool_free` 제외). Conv 블록, C3, SPPF, Upsample, Concat 등.
- **detect / decode / nms**: Detect Head 1×1 Conv 3스케일, 디코딩, NMS 각각.
- **backbone / neck / head / total**: 위 구간들을 묶은 합계와 전체 추론 시간.

### 보드(BARE_METAL): mcycle 기반 사이클 → ms

RISC-V **mcycle** CSR(0xB00)과 **mcycleh** CSR(0xB80)을 읽어 64비트 사이클 수를 만든다. 5분 이상 걸리는 구간도 32비트 오버플로 없이 측정한다.

**64비트 값 읽기** (`csrc/utils/mcycle.h`):

- 하위 32비트: `mcycle_read_lo()` → `csrr %0, 0xB00`
- 상위 32비트: `mcycle_read_hi()` → `csrr %0, 0xB80`
- **롤오버 방지**: `hi` → `lo` → `hi2` 순으로 읽고, `hi != hi2`이면 하위 32비트가 0xFFFFFFFF→0으로 넘어가는 순간으로 간주해 **`lo`를 한 번 더 읽어** `hi2`와 짝을 맞춘다. 그 후 `(hi2 << 32) | lo`를 반환해, 롤오버 직후에 값이 약 42초 뒤로 튀는 현상을 막는다.

```c
/* mcycle.h: hi → lo → hi2, hi != hi2 시 lo 재읽기 */
static inline uint64_t mcycle_read64(void) {
    uint32_t hi  = mcycle_read_hi();
    uint32_t lo  = mcycle_read_lo();
    uint32_t hi2 = mcycle_read_hi();
    if (hi != hi2)
        lo = mcycle_read_lo();
    return ((uint64_t)hi2 << 32) | (uint64_t)lo;
}
```

**구간 시간**: 구간 시작·끝에서 `timer_read64()`(= `mcycle_read64()`)를 부르고, `timer_delta64(start, end)` = `end - start`로 사이클 수를 얻는다.  
**ms 변환**: `cycles / (CPU_MHZ * 1000)` (예: 100MHz면 1ms = 100,000 사이클).  
보드용 `xil_printf`는 `%f`를 지원하지 않으므로 **정수 ms**만 출력한다 (`main.c`의 `LAYER_MS_INT`, `LAYER_LOG` 매크로).

### 호스트: 마이크로초 → ms

Windows는 `QueryPerformanceCounter` / `QueryPerformanceFrequency`, 그 외는 `gettimeofday`로 **마이크로초** 단위 타임스탬프를 읽는다 (`mcycle.h`의 `host_time_us()` → `timer_read64()`).  
구간은 `timer_delta64(start, end)`로 μs 차이를 구하고, 출력 시 `/1000.0`으로 **ms**로 보여 호스트·보드 결과를 같은 단위로 비교한다.

### main.c에서의 사용

- **타이머 읽기**: `t_layer = timer_read64();` → 연산 → `layer_cycles[i] = timer_delta64(t_layer, timer_read64());`
- **출력**: 레이어는 `LAYER_LOG(i, layer_cycles[i], &lN[0])`로 **각 레이어 통과 시마다** `  Ln xxx ms (0x........)` 한 줄 출력.  
  마지막에 `[mcycle]`(사이클 수)와 `[time @ 100MHz]`(또는 호스트 `[time]`)로 backbone/neck/head/decode/nms/total을 한 줄로 요약한다.

## 워크플로우 요약

1. **가중치**: `tools/export_weights_to_bin.py` → `assets/weights.bin`
2. **이미지 전처리**: `tools/preprocess_image_to_bin.py` → `data/input/preprocessed_image.bin`
3. **C 추론**: `./main` (호스트) 또는 보드에서 실행
4. **결과 확인**: `tools/decode_detections.py` 로 bin → txt/시각화

Bare-metal 보드에서는 가중치·이미지를 DDR에 미리 적재한 뒤 실행하며, 결과는 UART로 받아 `tools/recv_detections_uart.py` 등으로 저장 후 동일하게 디코딩 가능.

## 성능 최적화 (conv2d)

Conv2D는 D-Cache·메모리 대역폭을 줄이기 위해 다음을 적용했다.

- **타일링**: 출력을 8×8 타일로 나누어 한 타일 내에서 입력/가중치 재사용.
- **가중치 재사용**: 루프 순서 `ic → b → dh → dw → kh → kw`. 필터 하나를 한 번 로드해 64픽셀에 64회 재사용.
- **Strength reduction**: 가장 안쪽 루프(kw)에서 인덱스 곱셈 제거, `x_row++`/`w_row++` 포인터 증감만 사용.
- **패딩 분리**: 타일 전체가 안전 영역인지 한 번만 체크 → 64회 분기를 1회로 축소.
- **누적 버퍼**: `acc_ptr = &acc_buf[dh][dw][0]`, `acc_ptr[b] += contrib` 로 다차원 인덱싱 오버헤드 감소.

상세 개념·코드 설명은 **[docs/CONV2D_OPTIMIZATION.md](docs/CONV2D_OPTIMIZATION.md)** 참고.

## 기술 요약

- **Fused 모델**: Conv+BN → Conv+Bias로 흡수, BN 연산 제거
- **NCHW**: 모든 텐서가 Batch×Channel×Height×Width
- **Anchor-based**: P3/P4/P5 각 3앵커, 255ch = 3×85 (bbox+obj+80클래스)
- **HW 출력**: 12바이트/검출 (x,y,w,h, class_id, confidence 등), 상세는 `decode.h` 의 `hw_detection_t`

## 테스트

블록별 단위 테스트 및 벡터 생성 방법은 **[docs/TESTING.md](docs/TESTING.md)** 참고.

## 라이선스 / 참고

- YOLOv5 계열 모델·가중치 사용 시 Ultralytics 라이선스 확인
- 변경 이력: [CHANGELOG.md](CHANGELOG.md)
