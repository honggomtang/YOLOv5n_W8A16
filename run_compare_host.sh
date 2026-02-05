#!/bin/bash
# FP32(W32A32) vs W8A32 호스트에서 각각 실행 후 결과 비교
# 사용: ./run_compare_host.sh        -> W32A32 vs W8A32 비교 (2회 실행)
#       ./run_compare_host.sh w8a32  -> W8A32만 빌드 후 실행 (weights_w8.bin)
#       ./run_compare_host.sh w8a16  -> W8A16만 빌드 후 실행 (weights_w8.bin)

set -e
cd "$(dirname "$0")"
OUT=data/output
mkdir -p "$OUT"

# W8A32 소스 목록 (decode/nms는 공통)
BLOCKS_W8A32="csrc/blocks/conv_w8a32.c csrc/blocks/c3_w8a32.c csrc/blocks/decode.c csrc/blocks/detect_w8a32.c csrc/blocks/nms.c csrc/blocks/sppf_w8a32.c"
OPS_W8A32="csrc/operations/bottleneck_w8a32.c csrc/operations/concat_w8a32.c csrc/operations/conv2d_w8a32.c csrc/operations/maxpool2d_w8a32.c csrc/operations/silu_w8a32.c csrc/operations/upsample_w8a32.c"

# W8A16 소스 목록
BLOCKS_W8A16="csrc/blocks/conv_w8a16.c csrc/blocks/c3_w8a16.c csrc/blocks/decode.c csrc/blocks/detect_w8a16.c csrc/blocks/nms.c csrc/blocks/sppf_w8a16.c"
OPS_W8A16="csrc/operations/bottleneck_w8a16.c csrc/operations/concat_w8a16.c csrc/operations/conv2d_w8a16.c csrc/operations/maxpool2d_w8a16.c csrc/operations/silu_w8a16.c csrc/operations/upsample_w8a16.c"

UTILS="csrc/utils/feature_pool.c csrc/utils/image_loader.c csrc/utils/weights_loader.c csrc/utils/timing.c csrc/utils/uart_dump.c"

if [ "$1" = "w8a32" ]; then
  echo "=== W8A32만 빌드 및 실행 (INT8 가중치, FP32 활성화) ==="
  gcc -o main csrc/main.c $BLOCKS_W8A32 $OPS_W8A32 $UTILS -I. -Icsrc -lm -std=c99 -O2 -DUSE_WEIGHTS_W8 2>&1
  ./main 2>&1 | tee "$OUT/w8a32_log.txt"
  echo "  저장: $OUT/detections.bin, $OUT/detections.txt, $OUT/w8a32_log.txt"
  echo "  (weights: assets/weights_w8.bin)"
  exit 0
fi

if [ "$1" = "w8a16" ]; then
  echo "=== W8A16 빌드 및 실행 ==="
  gcc -o main csrc/main.c $BLOCKS_W8A16 $OPS_W8A16 $UTILS -I. -Icsrc -lm -std=c99 -O2 -DUSE_W8A16 -DUSE_WEIGHTS_W8 2>&1
  ./main 2>&1 | tee "$OUT/w8a16_log.txt"
  echo "  저장: $OUT/detections.bin, $OUT/detections.txt, $OUT/w8a16_log.txt"
  echo "  (weights: assets/weights_w8.bin 필요)"
  exit 0
fi

echo "=== 1) W32A32 (FP32 가중치, weights.bin) 빌드 및 실행 ==="
gcc -o main csrc/main.c $BLOCKS_W8A32 $OPS_W8A32 $UTILS -I. -Icsrc -lm -std=c99 -O2 2>&1
./main 2>&1 | tee "$OUT/ref_fp32_log.txt"
cp -f "$OUT/detections.bin" "$OUT/ref_fp32_detections.bin"
cp -f "$OUT/detections.txt" "$OUT/ref_fp32_detections.txt"
echo "  저장: $OUT/ref_fp32_detections.bin, ref_fp32_log.txt"

echo ""
echo "=== 2) W8A32 (INT8 가중치, weights_w8.bin) 빌드 및 실행 ==="
gcc -o main csrc/main.c $BLOCKS_W8A32 $OPS_W8A32 $UTILS -I. -Icsrc -lm -std=c99 -O2 -DUSE_WEIGHTS_W8 2>&1
./main 2>&1 | tee "$OUT/w8_log.txt"
echo "  저장: $OUT/detections.bin (W8A32), $OUT/w8_log.txt"

echo ""
echo "=== 3) 비교 ==="
python3 tools/compare_fp32_w8.py --out-dir "$OUT"
