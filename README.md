# YOLOv5n Pure C Implementation (W8A32 / W8A16)

순수 C로 구현한 YOLOv5n(nano) 객체 탐지 추론 엔진. 외부 라이브러리 없이 동작하며, 호스트 빌드와 **Bare-metal(FPGA)** 빌드를 하나의 코드베이스로 지원. **W8A16 경로**는 Conv 하드웨어 가속기(RTL) 및 드라이버와 연동 가능.

## 목표

- **최종**: MicroBlaze V(RISC-V) 등 FPGA에서 YOLOv5n 추론 실행
- **제약**: OpenCV/OpenBLAS 등 미사용, 순수 C만 사용
- **상태**: Python YOLOv5n과 동일한 추론 결과 (호스트·보드 검증 완료)
- **구성**: **W8A32**(INT8 가중치 + FP32 활성화), **W8A16**(INT8 가중치 + INT16 Q6.10 활성화) 두 경로 지원. W8A16은 `-DUSE_CONV_ACC`로 Conv 하드웨어 가속 시도(제약 미충족 시 SW 폴백). Decode/NMS는 공통(float 입력).

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
│   ├── types_w8a16.h, types_w8a32.h
│   │
│   ├── drivers/                # 하드웨어 가속기 드라이버 (USE_CONV_ACC)
│   │   └── conv_acc_driver.c,h # Conv 가속기 GPIO/DMA 제어
│   │
│   ├── blocks/                 # 고수준 블록 (W8A32 / W8A16 분리)
│   │   ├── conv_w8a32.c,h, conv_w8a16.c,h
│   │   ├── c3_w8a32.c,h, c3_w8a16.c,h
│   │   ├── sppf_w8a32.c,h, sppf_w8a16.c,h
│   │   ├── detect_w8a32.c,h, detect_w8a16.c,h
│   │   ├── decode.c,h, nms.c,h
│   │
│   ├── operations/             # 저수준 연산 (W8A32 / W8A16 분리)
│   │   ├── conv2d_w8a32.c,h, conv2d_w8a16.c,h
│   │   ├── silu_w8a32.c,h, silu_w8a16.c,h + silu_lut_data.h
│   │   ├── bottleneck, concat, maxpool2d, upsample
│   │
│   └── utils/
│       ├── weights_loader.c,h, image_loader.c,h, feature_pool.c,h
│       ├── mcycle.h, timing.c,h, uart_dump.c,h
│
├── vsrc/                       # Conv 가속기 RTL (Verilog)
│   ├── conv_acc_top.v          # 탑 모듈
│   ├── conv_acc_buffer.v       # 라인 버퍼
│   ├── conv_acc_compute.v      # PE 클러스터 오케스트레이션
│   ├── conv_acc_requant.v      # Requant
│   ├── pe_cluster.v, pe_mac.v   # MAC 연산 유닛
│   ├── tb_conv_acc.v           # 테스트벤치
│   └── run_tb_conv_acc.bat     # iverilog 시뮬레이션
│
├── data/                       # 입력·전처리·결과
├── tools/
│   ├── export_weights_to_bin.py, export_acc_repack_from_w8.py  # 가속기용 repack
│   ├── preprocess_image_to_bin.py, preprocess_image_a16.py
│   ├── gen_silu_lut.py, run_python_yolov5n_fused.py
│   ├── compare_fp32_w8.py, verify_weights_bin.py
│   ├── recv_detections_uart.py, uart_to_detections_txt.py
│   └── ...
├── tests/                      # 단위 테스트
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
| **W8A16+Conv 가속** | `-O2 -DUSE_W8A16 -DUSE_WEIGHTS_W8 -DUSE_CONV_ACC` | weights_w8.bin (+conv_acc_driver.c) |

**스크립트** (`run_compare_host.sh`)

```bash
./run_compare_host.sh           # W32A32 vs W8A32 비교 (2회 실행 후 compare)
./run_compare_host.sh w8a32     # W8A32만 빌드·실행
./run_compare_host.sh w8a16     # W8A16만 빌드·실행
```

W8A16 빌드 시 소스: `csrc/main.c` + `csrc/blocks/*` + `csrc/operations/*` + `csrc/utils/*`. Conv 가속 사용 시 `csrc/drivers/conv_acc_driver.c` 추가.

**실행**

```bash
./main
```

- 입력: `data/input/preprocessed_image.bin` (W8A32/FP32) 또는 W8A16 호스트 시 `preprocessed_image_a16.bin` (생성: `tools/preprocess_image_a16.py --from-float data/input/preprocessed_image.bin --out data/input/preprocessed_image_a16.bin`)
- 출력: `data/output/detections.bin`, `detections.txt`
- 콘솔: 레이어별 시간 `L0 ... ms (0x....)`, 마지막에 `[time] backbone=... neck=... head=...` 요약

**Windows (PowerShell)**  
- 빌드: `build_host.bat w8a16`  
- 실행: `.\main.exe` (현재 디렉터리 실행 파일은 `.\` 필요)

### Bare-metal (Vitis)

- `-DBARE_METAL`, DDR에서 이미지·가중치 로드, UART로 결과 덤프.  
- **W8A16 시 DDR 로드**  
  - 이미지: `preprocessed_image_a16.bin` → **0x8F000000** (24B 헤더 + int16 Q6.10, zero-copy L0 입력).  
  - 가중치: `weights_w8.bin` → **0x88000000**.  
  - 피처 풀: **48MB** (0x82000000~). Detect 출력 버퍼(p3/p4/p5)는 **DETECT_HEAD_BASE** 고정 영역 사용(힙 미사용).  
- XSCT 예: `dow -data "path/preprocessed_image_a16.bin" 0x8F000000` → `dow -data "path/weights_w8.bin" 0x88000000` → `dow path/app.elf` → `con`.  
- 상세: `platform_config.h`의 DDR 맵, `csrc/drivers/conv_acc_driver.h`의 GPIO 베이스 주소 참고.

## 단계별 시간 측정

- 레이어 L0~L23, Detect(L24), Decode, NMS를 ms 단위로 측정.
- 호스트: 마이크로초 기반 → ms 출력. 보드: mcycle → `cycles/(CPU_MHZ*1000)` ms.
- W8A16: 각 레이어 통과 시 `L%d ... ms (0x%04X int) (0x%08X fp) (ref: 0x%08X)` 형태로 출력 (ref는 Q6.10→float 비교용).

## 기술 요약

- **Fused**: Conv+BN → Conv+Bias 흡수.
- **NCHW**, **Anchor-based**: P3/P4/P5 각 3앵커, 255ch = 3×85.
- **HW 출력**: 12바이트/검출 (decode.h `hw_detection_t`).
- **W8A16 가중치 4-way pack**: Conv 가중치 [OC,IC,KH,KW]를 로드 후 [OC_padded/4, IC, KH, KW]로 repack. conv2d는 uint32_t 단위 1회 로드로 4채널 누산.
- **W8A16 입력 zero-copy**: BARE_METAL에서는 DDR에 `preprocessed_image_a16.bin`(24B 헤더 + int16)을 넣고, L0 입력을 복사 없이 해당 주소로 사용.
- **Conv 가속기**: vsrc RTL(pe_mac, pe_cluster, conv_acc_buffer, conv_acc_compute, conv_acc_requant). 3×3/1×1 Conv 지원, 제약(c_in 짝수, 라인 버퍼 3072, 가중치 슬롯 2048 등) 미충족 시 SW 폴백.

## Conv 가속기 RTL (vsrc)

- **conv_acc_top.v**: AXI/GPIO 인터페이스, 탑 모듈.
- **conv_acc_buffer.v**: 라인 버퍼(MAX_W=3072).
- **conv_acc_compute.v**: PE 클러스터 오케스트레이션(8 clusters × 32 PE).
- **conv_acc_requant.v**: 누산 → requant → int16 출력.
- 시뮬레이션: `iverilog` 또는 Vivado. `vsrc/run_tb_conv_acc.bat` (Windows).

## 라이선스 / 참고

- YOLOv5 계열 모델·가중치 사용 시 Ultralytics 라이선스 확인
- 변경 이력: [CHANGELOG.md](CHANGELOG.md)
