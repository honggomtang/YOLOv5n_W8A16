# YOLOv5n Pure C Implementation (W8A32 / W8A16)

순수 C로 구현한 YOLOv5n(nano) 객체 탐지 추론 엔진. 외부 라이브러리 없이 동작하며, 호스트 빌드와 **Bare-metal(FPGA)** 빌드를 하나의 코드베이스로 지원한다.

## 목표

- **최종**: MicroBlaze V(RISC-V) 등 FPGA에서 YOLOv5n 추론 실행
- **제약**: OpenCV/OpenBLAS 등 미사용, 순수 C만 사용
- **상태**: Python YOLOv5n과 동일한 추론 결과 (호스트·보드 검증 완료)
- **구성**: **W8A32**(INT8 가중치 + FP32 활성화), **W8A16**(INT8 가중치 + INT16 Q6.10 활성화) 두 경로 지원. Decode/NMS는 공통(float 입력).

## 폴더 구조

```
YOLOv5n_W8A16/
├── assets/
│   ├── yolov5n.pt              # PyTorch 원본
│   ├── weights.bin             # FP32 가중치 (W8A32 FP32 경로)
│   └── weights_w8.bin          # INT8 가중치 (W8A32/W8A16 공용, scale 내장)
│
├── csrc/
│   ├── main.c                  # 추론 파이프라인 (USE_W8A16 시 W8A16 경로)
│   ├── platform_config.h       # BARE_METAL DDR 맵 / CPU_MHZ
│   ├── types_w8a16.h          # W8A16 Q6.10 타입·헬퍼
│   ├── types_w8a32.h          # W8A32 타입
│   │
│   ├── blocks/                 # 고수준 블록 (W8A32 / W8A16 분리)
│   │   ├── conv_w8a32.c,h, conv_w8a16.c,h
│   │   ├── c3_w8a32.c,h, c3_w8a16.c,h
│   │   ├── sppf_w8a32.c,h, sppf_w8a16.c,h
│   │   ├── detect_w8a32.c,h, detect_w8a16.c,h
│   │   ├── decode.c,h          # 공통 (float 입력)
│   │   └── nms.c,h             # 공통
│   │
│   ├── operations/             # 저수준 연산 (W8A32 / W8A16 분리)
│   │   ├── conv2d_w8a32.c,h, conv2d_w8a16.c,h
│   │   ├── silu_w8a32.c,h, silu_w8a16.c,h + silu_lut_data.h  # W8A16: LUT
│   │   ├── bottleneck_w8a32.c,h, bottleneck_w8a16.c,h
│   │   ├── concat_w8a32.c,h, concat_w8a16.c,h
│   │   ├── maxpool2d_w8a32.c,h, maxpool2d_w8a16.c,h
│   │   └── upsample_w8a32.c,h, upsample_w8a16.c,h
│   │
│   └── utils/
│       ├── weights_loader.c,h  # weights.bin / weights_w8.bin, W8A16 시 4-way pack(repack)
│       ├── image_loader.c,h
│       ├── feature_pool.c,h    # scratch_alloc (W8A16 피처맵 풀)
│       ├── mcycle.h            # 타이머 (mcycle / 호스트)
│       ├── timing.c,h          # 레이어·연산별 시간
│       └── uart_dump.c,h       # BARE_METAL UART 덤프
│
├── data/
│   ├── image/, input/, output/ # 입력·전처리·결과
│   └── output/ref/             # Python 참조 결과
│
├── tools/
│   ├── export_weights_to_bin.py
│   ├── preprocess_image_to_bin.py, preprocess_image_a16.py
│   ├── gen_silu_lut.py         # W8A16 SiLU LUT 헤더 생성
│   ├── run_python_yolov5n_fused.py, decode_detections.py
│   ├── compare_fp32_w8.py, verify_weights_bin.py, reweight_align4.py
│   ├── recv_detections_uart.py, uart_to_detections_txt.py
│   └── gen_test_vectors.py, compute_layer_multipliers.py, compute_layer_shifts.py
│
├── tests/                      # 단위 테스트 (test_*.c, test_vectors_*.h)
├── docs/                       # 상세 문서
│   ├── W8A16_IMPLEMENTATION.md # W8A16 구현 정리 (개념·코드)
│   ├── W8A32_IMPLEMENTATION.md
│   ├── CONV2D_OPTIMIZATION.md, DATA_CACHE_USAGE.md, VITIS_BUILD.md
│   └── TESTING.md
├── CHANGELOG.md
└── README.md
```

## 빌드 및 실행

### 호스트

**준비**

- Python 도구: `pip install -r requirements.txt`
- 전처리 이미지: `tools/preprocess_image_to_bin.py` → `data/input/preprocessed_image.bin`
- 가중치: `tools/export_weights_to_bin.py` → `assets/weights.bin`  
  W8 경로: INT8 가중치 `assets/weights_w8.bin` (별도 양자화 도구로 생성)

**빌드**

| 대상 | 옵션 | 가중치 |
|------|------|--------|
| W32A32 (FP32) | `-O2 -I. -Icsrc` | weights.bin |
| W8A32 | `-O2 -DUSE_WEIGHTS_W8` | weights_w8.bin |
| **W8A16** | `-O2 -DUSE_W8A16 -DUSE_WEIGHTS_W8` | weights_w8.bin |

**스크립트** (`run_compare_host.sh`)

```bash
./run_compare_host.sh           # W32A32 vs W8A32 비교 (2회 실행 후 compare)
./run_compare_host.sh w8a32     # W8A32만 빌드·실행
./run_compare_host.sh w8a16     # W8A16만 빌드·실행
```

W8A16 빌드 시 소스: `csrc/main.c` + `csrc/blocks/conv_w8a16.c c3_w8a16.c decode.c detect_w8a16.c nms.c sppf_w8a16.c` + `csrc/operations/bottleneck_w8a16.c concat_w8a16.c conv2d_w8a16.c maxpool2d_w8a16.c silu_w8a16.c upsample_w8a16.c` + `csrc/utils/feature_pool.c image_loader.c weights_loader.c timing.c uart_dump.c`

**실행**

```bash
./main
```

- 입력: `data/input/preprocessed_image.bin`
- 출력: `data/output/detections.bin`, `detections.txt`
- 콘솔: 레이어별 시간 `L0 ... ms (0x....)`, 마지막에 `[time] backbone=... neck=... head=...` 요약

### Bare-metal (Vitis)

- `-DBARE_METAL`, DDR에서 이미지·가중치 로드, UART로 결과 덤프.  
- 상세: [docs/VITIS_BUILD.md](docs/VITIS_BUILD.md), [docs/DATA_CACHE_USAGE.md](docs/DATA_CACHE_USAGE.md)

## 단계별 시간 측정

- 레이어 L0~L23, Detect(L24), Decode, NMS를 ms 단위로 측정.
- 호스트: 마이크로초 기반 → ms 출력. 보드: mcycle → `cycles/(CPU_MHZ*1000)` ms.
- W8A16: 각 레이어 통과 시 `L%d ... ms (0x%04X int) (0x%08X fp) (ref: 0x%08X)` 형태로 출력 (ref는 Q6.10→float 비교용).

## 기술 요약

- **Fused**: Conv+BN → Conv+Bias 흡수.
- **NCHW**, **Anchor-based**: P3/P4/P5 각 3앵커, 255ch = 3×85.
- **HW 출력**: 12바이트/검출 (decode.h `hw_detection_t`).
- **W8A16 가중치 4-way pack**: Conv 가중치 [OC,IC,KH,KW]를 로드 후 [OC_padded/4, IC, KH, KW]로 repack(같은 ic,kh,kw에 대한 OC 4개를 uint32_t 하나로). OC가 4의 배수가 아니면(예: Detect 255) 0 패딩. 패킹 버퍼는 4바이트 정렬 할당(aligned_alloc 또는 BARE_METAL용 정렬 할당자 권장). conv2d는 uint32_t 단위 1회 로드로 4채널 누산.

## 문서

- **W8A16 구현 전체 정리**: [docs/W8A16_IMPLEMENTATION.md](docs/W8A16_IMPLEMENTATION.md) — 데이터 형식, Conv/Requant/Bias, **가중치 4-way pack(repack)·정렬**, SiLU LUT, 블록 API(로더+이름), main 흐름, 빌드.
- W8A32: [docs/W8A32_IMPLEMENTATION.md](docs/W8A32_IMPLEMENTATION.md)
- Conv2D 최적화: [docs/CONV2D_OPTIMIZATION.md](docs/CONV2D_OPTIMIZATION.md)
- 테스트: [docs/TESTING.md](docs/TESTING.md)

## 라이선스 / 참고

- YOLOv5 계열 모델·가중치 사용 시 Ultralytics 라이선스 확인
- 변경 이력: [CHANGELOG.md](CHANGELOG.md)
