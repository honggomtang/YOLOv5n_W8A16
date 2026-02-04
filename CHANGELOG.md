# CHANGELOG

## Unreleased

- **W8A32**: INT8 weights + FP32 compute 경로 정리 (SPPF 포함, dequant 풀 제거)
- **conv2d_w8**: local_w pre-load(형변환 비용 절감), 32-bit bundle load(정렬 시), 1×1 fast path 추가
- **Windows 호스트 빌드**: `build_host.bat w8` 옵션 추가

