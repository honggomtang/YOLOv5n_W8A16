# CHANGELOG

## Unreleased

- **W8A16 zero-copy a16 입력**: BARE_METAL에서 DDR `preprocessed_image_a16.bin`(24B 헤더 + int16)을 L0 입력으로 복사 없이 사용. `image_init_from_memory_a16`, `image_load_from_bin_a16` 추가. `yolov5n_inference_w8a16`에 `x0_a16_zero_copy` 인자 추가.
- **W8A16 BARE_METAL 메모리**: p3/p4/p5를 `DETECT_HEAD_BASE` 고정 DDR 사용(힙 할당 제거). `FEATURE_POOL_SIZE`를 USE_W8A16 시 48MB로 확대. `WEIGHTS_W8_DDR_SIZE` 8MB, `IMAGE_A16_DDR_SIZE` 정의.
- **W8A16 Conv2D 32비트 로드**: 활성화 x를 `uint32_t`로 2개씩 읽도록 1x1(dw 2 언롤), 일반 경로(kw 2 언롤) 최적화. 정렬/홀수 예외 처리.
- **가중치 로드**: `parse_weights_w8`에서 num_tensors 0 또는 >512 검사. BARE_METAL 실패 시 0x88000000 첫 워드 디버그 로그.
- **빌드**: W8A32 전용 헤더를 `#ifndef USE_W8A16`로 감싸 링크 오류 방지. Windows `build_host.bat w8a16`에서 `-DUSE_W8A16`을 gcc 인자 앞에 배치.
- **W8A32**: INT8 weights + FP32 compute 경로 정리 (SPPF 포함, dequant 풀 제거)
- **conv2d_w8**: local_w pre-load(형변환 비용 절감), 32-bit bundle load(정렬 시), 1×1 fast path 추가
- **Windows 호스트 빌드**: `build_host.bat w8` 옵션 추가

