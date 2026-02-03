#!/bin/bash
# FP32(수정 전) vs W8A32(수정 후) 호스트에서 각각 실행 후 결과 비교
# 사용: ./run_compare_host.sh   (프로젝트 루트에서)

set -e
cd "$(dirname "$0")"
OUT=data/output
mkdir -p "$OUT"

echo "=== 1) FP32 (수정 전) 빌드 및 실행 ==="
gcc -o main csrc/main.c csrc/blocks/*.c csrc/operations/*.c csrc/utils/*.c -I. -Icsrc -lm -std=c99 -O2 2>&1
./main 2>&1 | tee "$OUT/ref_fp32_log.txt"
cp -f "$OUT/detections.bin" "$OUT/ref_fp32_detections.bin"
cp -f "$OUT/detections.txt" "$OUT/ref_fp32_detections.txt"
echo "  저장: $OUT/ref_fp32_detections.bin, ref_fp32_log.txt"

echo ""
echo "=== 2) W8A32 (수정 후) 빌드 및 실행 ==="
gcc -o main csrc/main.c csrc/blocks/*.c csrc/operations/*.c csrc/utils/*.c -I. -Icsrc -lm -std=c99 -O2 -DUSE_WEIGHTS_W8 2>&1
./main 2>&1 | tee "$OUT/w8_log.txt"
echo "  저장: $OUT/detections.bin (W8), $OUT/w8_log.txt"

echo ""
echo "=== 3) 비교 ==="
python3 tools/compare_fp32_w8.py --out-dir "$OUT"
