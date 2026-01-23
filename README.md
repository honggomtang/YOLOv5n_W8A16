# YOLOv5n 순수 C 포팅 프로젝트

전기전자공학부/임베디드(특히 MicroBlaze V / Vitis) 환경에서 실행 가능하도록,
PyTorch YOLOv5n 모델을 **순수 C 언어**로 포팅한 프로젝트입니다.

## 현재 상태

✅ **전체 파이프라인 구현 완료**
- Layer 0~23: 특징 추출 파이프라인 (모든 레이어 Python과 동일, diff < 1e-4)
- Layer 24: Detect Head (cv2, cv3 브랜치 구현 완료, 검증 통과)
- Decode: DFL 기반 디코딩 구현 완료 (검증 통과)
- **NMS**: Non-Maximum Suppression 구현 완료 (Python 결과와 일치, 검증 통과)
- **.bin 파일 지원 완료**: 모든 테스트가 .bin 파일 사용 (헤더 파일 제거됨)
  - `weights.bin`: 가중치 바이너리 (10MB)
  - `preprocessed_image.bin`: 전처리된 이미지 바이너리 (4.7MB)
  - 개발/테스트 및 임베디드 환경 모두 .bin 파일 사용

## 폴더 구조

```
yolov5n/
├── assets/              # 모델 파일 및 가중치
│   ├── yolov5n.pt      # PyTorch 모델 파일
│   └── weights.bin      # 바이너리 가중치 (10MB, 모든 환경에서 사용)
├── data/
│   ├── image/           # 입력 이미지
│   └── input/
│       └── preprocessed_image.bin  # 전처리된 이미지 바이너리 (4.7MB, 모든 환경에서 사용)
├── tools/               # Python 유틸리티
│   ├── export_weights_to_bin.py      # .pt → weights.bin 변환
│   ├── preprocess_image_to_bin.py    # 이미지 → .bin 변환
│   ├── gen_*_test_vectors.py         # 테스트 벡터 생성기
│   └── ...
├── csrc/                # C 소스 코드
│   ├── operations/      # 기본 연산 (Conv2D, BN+SiLU, Concat, Upsample 등)
│   ├── blocks/          # 복합 블록 (Conv, C3, SPPF, Detect, Decode)
│   └── utils/           # 유틸리티 (weights_loader, image_loader)
├── tests/               # 테스트 코드 및 테스트 벡터
│   ├── test_*.c         # 단위 테스트
│   └── test_vectors_*.h # Python에서 생성한 참조 데이터
└── Makefile            # 빌드 자동화
```

## 구현된 블록 및 연산

### Operations (기본 연산)
- **Conv2D**: 2D Convolution (NCHW)
- **BN+SiLU**: Batch Normalization + SiLU Activation (fused)
- **Bottleneck**: 1x1 Conv → 3x3 Conv + Residual
- **Concat**: Channel-wise concatenation (2개/4개 지원)
- **MaxPool2D**: 2D Max Pooling
- **Upsample**: Nearest neighbor ×2 upsampling

### Blocks (복합 블록)
- **Conv**: Conv2D + BN + SiLU
- **C3**: Cross-stage partial bottleneck (다중 Bottleneck 지원)
- **SPPF**: Spatial Pyramid Pooling Fast
- **Detect**: Detect Head (cv2: bbox prediction, cv3: class prediction)
- **Decode**: DFL 기반 디코딩 (bbox + confidence + class)
- **NMS**: Non-Maximum Suppression (중복 detection 제거)

## 빠른 시작

### 1. 환경 설정

```bash
cd /Users/kinghong/Desktop/yolov5n
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy pillow ultralytics
```

### 2. 가중치 및 이미지 .bin 파일 생성

#### 가중치 .bin 파일 생성

```bash
source .venv/bin/activate
python tools/export_weights_to_bin.py \
    --pt assets/yolov5n.pt \
    --out assets/weights.bin \
    --trust-pickle
```

#### 이미지 .bin 파일 생성

```bash
source .venv/bin/activate
python tools/preprocess_image_to_bin.py \
    --img data/image/zidane.jpg \
    --out data/input/preprocessed_image.bin \
    --size 640
```

> **참고**: 
> - PyTorch 2.6+에서는 `--trust-pickle` 플래그가 필요할 수 있습니다.
> - `.bin` 파일은 컴파일 시간 단축 및 메모리 효율성 향상에 유리합니다.
> - 개발/테스트 및 임베디드 환경 모두 `.bin` 파일을 사용합니다.

### 3. 단위 테스트 실행

#### Conv Block 테스트
```bash
make test_conv0
```

#### C3 Block 테스트
```bash
make test_c3
```

#### SPPF Block 테스트
```bash
make test_sppf_gen  # 테스트 벡터 생성
make test_sppf      # 테스트 실행
```

#### Upsample 테스트
```bash
make test_upsample_gen
make test_upsample
```

### 4. 전체 파이프라인 검증

#### Layer 0~9 검증 (이미지 입력)
```bash
make test_layer0_9_gen  # zidane.jpg 사용 (64x64로 리사이즈)
make test_layer0_9
```

#### Layer 0~23 검증 (랜덤 입력)
```bash
make test_layer0_23_gen  # 32x32 랜덤 이미지
make test_layer0_23
```

**예상 출력:**
```
Layer 0 diff = 5.72205e-06 OK
Layer 1 diff = 1.90735e-05 OK
...
Layer 23 diff = 0 OK

All layers OK
```

#### Detect Head 테스트
```bash
# Detect 테스트 벡터 생성
source .venv/bin/activate
python tools/gen_detect_test_vectors.py \
    --pt assets/yolov5n.pt \
    --img data/image/zidane.jpg \
    --size 32 \
    --out tests/test_vectors_detect.h

# Detect 테스트 실행
gcc -o tests/test_detect tests/test_detect.c csrc/blocks/detect.c \
    csrc/blocks/conv.c csrc/operations/conv2d.c csrc/operations/bn_silu.c \
    csrc/utils/weights_loader.c -I. -Icsrc -Iassets -Itests -lm -std=c99
./tests/test_detect
```

#### Decode 테스트
```bash
# Decode 테스트 벡터 생성
source .venv/bin/activate
python tools/gen_decode_test_vectors.py \
    --pt assets/yolov5n.pt \
    --size 32 \
    --out tests/test_vectors_decode.h \
    --conf-threshold 0.01

# Decode 테스트 실행
gcc -o tests/test_decode tests/test_decode.c csrc/blocks/decode.c \
    -I. -Icsrc -Iassets -Itests -lm -std=c99
./tests/test_decode
```

#### NMS 테스트
```bash
# NMS 테스트 벡터 생성
source .venv/bin/activate
python tools/gen_nms_full_test_vectors.py \
    --pt assets/yolov5n.pt \
    --img data/image/zidane.jpg \
    --out tests/test_vectors_nms_full.h \
    --conf 0.25 --iou 0.45

# NMS 테스트 실행
gcc -o tests/test_nms_full tests/test_nms_full.c csrc/blocks/nms.c \
    -I. -Icsrc -Iassets -Itests -lm -std=c99
./tests/test_nms_full
```

**예상 출력:**
```
✅ All detections matched (tolerance=0.01)
Python: 4 detections
C NMS:  4 detections
```

## 상세 가이드

### .bin 파일 형식

가중치와 이미지는 바이너리 형식으로 저장됩니다:

- **weights.bin**: 텐서 개수 + 각 텐서(이름, shape, 데이터)
- **preprocessed_image.bin**: 원본 크기, 스케일, 패딩 정보 + 이미지 데이터

C 코드에서는 `weights_loader`와 `image_loader` 유틸리티를 사용하여 로드합니다.

### 레이어 구조

YOLOv5n은 다음과 같은 레이어 구조를 가집니다:

- **Layer 0~9**: Backbone (특징 추출)
  - Conv, C3, SPPF 블록들
- **Layer 10~17**: Neck (FPN/PAN 구조)
  - Upsample, Concat, C3 블록들
- **Layer 18~23**: Head (Detection 준비)
  - Conv, Concat, C3 블록들
- **Layer 24**: Detect Head (구현 완료)
  - cv2: bbox prediction (64채널)
  - cv3: class prediction (80채널)
- **Decode**: DFL 기반 디코딩 (구현 완료)
- **NMS**: Non-Maximum Suppression (구현 완료)

### Concat 참조 관계

- **Layer 12**: Concat([Layer 11, Layer 6])
- **Layer 16**: Concat([Layer 15, Layer 4])
- **Layer 19**: Concat([Layer 18, Layer 14])
- **Layer 22**: Concat([Layer 21, Layer 10])

## MicroBlaze/Vitis 통합

### .bin 파일 사용 (임베디드용, **권장**)

임베디드 환경에서는 파일 시스템 없이 **메모리 직접 접근** 방식으로 가중치와 이미지를 사용합니다.

#### 1. Linker Script 설정

Vivado/Vitis에서 DDR3 메모리의 특정 영역을 예약:

```
MEMORY
{
  ...
  WEIGHTS_MEM : ORIGIN = 0x81000000, LENGTH = 0x02400000  /* 36MB 가중치 */
  IMAGE_MEM   : ORIGIN = 0x83400000, LENGTH = 0x01400000  /* 20MB 이미지 */
  ...
}
```

#### 2. XSCT로 메모리 로드

보드 시작 시 XSCT로 .bin 파일을 메모리에 직접 로드:

```tcl
# XSCT 스크립트 예시
connect
targets -set -filter {name =~ "*MicroBlaze*"}
dow assets/weights.bin 0x81000000
dow data/input/preprocessed_image.bin 0x83400000
```

#### 3. C 코드에서 사용

```c
#include "csrc/utils/weights_loader.h"
#include "csrc/utils/image_loader.h"

#define WEIGHTS_BASE_ADDR  0x81000000
#define IMAGE_BASE_ADDR    0x83400000

int main(void) {
    weights_loader_t weights;
    preprocessed_image_t img;
    
    // 메모리 직접 접근 (파일 시스템 없음)
    weights_init_from_memory(WEIGHTS_BASE_ADDR, &weights);
    image_init_from_memory(IMAGE_BASE_ADDR, &img);
    
    // 가중치 사용
    const tensor_info_t* conv0_w = weights_find_tensor(&weights, "model.0.conv.weight");
    // ...
    
    weights_free(&weights);
    image_free(&img);
    return 0;
}
```

### 개발/테스트 환경에서 사용

```c
#include "csrc/utils/weights_loader.h"
#include "csrc/utils/image_loader.h"

weights_loader_t weights;
preprocessed_image_t img;

// 파일 시스템에서 로드
weights_load_from_file("assets/weights.bin", &weights);
image_load_from_bin("data/input/preprocessed_image.bin", &img);

// 가중치 사용
const float* conv0_w = weights_get_tensor_data(&weights, "model.0.conv.weight");
// ...

weights_free(&weights);
image_free(&img);
```

### .bin 파일의 장점

1. **컴파일 시간 단축**: 대용량 헤더 파일을 컴파일에 포함하지 않음
2. **메모리 효율**: 런타임에 필요한 데이터만 메모리에 로드
3. **파일 시스템 불필요**: 임베디드 환경에서 파일 시스템 오버헤드 없음
4. **빠른 접근**: 메모리 직접 접근으로 최고 속도

### 최적화 팁

- **메모리**: 큰 배열(가중치, 중간 버퍼)을 외부 메모리/Flash에 배치
- **속도**: INT8 양자화, 레이어별 선택적 추출
- **정확도**: Float32 기본 사용 (필요시 Float16 고려)


## 다음 단계

- [x] Layer 0~23 구현 및 검증
- [x] Layer 24 (Detect Head) 구현 및 검증
- [x] Decode 블록 구현 및 검증
- [x] NMS (Non-Maximum Suppression) 구현 및 검증
- [x] .bin 파일 지원 완료 (모든 테스트가 .bin 파일 사용)
- [x] weights_loader, image_loader 유틸리티 구현
- [ ] 전체 파이프라인 통합 테스트 (main.c)
- [ ] MicroBlaze V 하드웨어 검증

## 라이선스

이 프로젝트는 YOLOv5n 모델을 C로 포팅한 구현체입니다.
원본 YOLOv5는 Ultralytics의 라이선스를 따릅니다.
