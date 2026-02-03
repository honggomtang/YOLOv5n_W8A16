#!/usr/bin/env python3
"""FP32(수정 전) vs W8A32(수정 후) 호스트 추론 결과 비교."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

# decode_detections.py와 동일 포맷
DET_RECORD_SIZE = 12  # <HHHHBBBB
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def read_detections_bin(path: Path) -> list[tuple]:
    """detections.bin → (x, y, w, h, class_id, confidence) 리스트."""
    if not path.exists():
        return []
    out = []
    with open(path, 'rb') as f:
        count = struct.unpack('B', f.read(1))[0]
        for _ in range(count):
            data = f.read(DET_RECORD_SIZE)
            if len(data) < DET_RECORD_SIZE:
                break
            x, y, w, h, cls_id, conf, _, _ = struct.unpack('<HHHHBBBB', data)
            out.append((x, y, w, h, cls_id, conf / 255.0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="FP32 vs W8A32 detections.bin 비교")
    ap.add_argument("--fp32", default=None, help="FP32 결과 detections.bin (수정 전)")
    ap.add_argument("--w8", default=None, help="W8A32 결과 detections.bin (수정 후)")
    ap.add_argument("--out-dir", default=None, help="기본 경로: data/output")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else root / "data" / "output"
    fp32_path = Path(args.fp32) if args.fp32 else out_dir / "ref_fp32_detections.bin"
    w8_path = Path(args.w8) if args.w8 else out_dir / "detections.bin"

    fp32 = read_detections_bin(fp32_path)
    w8 = read_detections_bin(w8_path)

    if not fp32_path.exists():
        print(f"FP32 결과 없음: {fp32_path}")
        print("  먼저 run_compare_host.sh (또는 수동으로 FP32 빌드/실행 후 ref_fp32_detections.bin 저장) 실행")
        return 1
    if not w8_path.exists():
        print(f"W8 결과 없음: {w8_path}")
        print("  W8A32 빌드/실행 후 data/output/detections.bin 생성")
        return 1

    print("=== FP32 (수정 전) vs W8A32 (수정 후) 비교 ===\n")
    print(f"  FP32: {fp32_path.name}  →  {len(fp32)} detections")
    print(f"  W8:   {w8_path.name}    →  {len(w8)} detections")
    print()

    # 요약
    print("--- 요약 ---")
    print(f"  개수: FP32={len(fp32)}, W8={len(w8)}, diff={len(w8) - len(fp32)}")
    if len(fp32) != len(w8):
        print("  [차이] 검출 개수 다름")
    print()

    # 항목별 비교 (최대 개수만큼)
    n = max(len(fp32), len(w8))
    if n == 0:
        print("  (검출 없음)")
        return 0

    print("--- 항목별 비교 (class conf% x y w h) ---")
    print(f"  {'#':>2}  {'FP32':<45}  {'W8':<45}  일치")
    print("  " + "-" * 100)
    for i in range(n):
        a = fp32[i] if i < len(fp32) else None
        b = w8[i] if i < len(w8) else None
        if a:
            x, y, w, h, cid, conf = a
            name = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else f"c{cid}"
            s_fp32 = f"{name} {conf*100:.0f}% ({x},{y},{w},{h})"
        else:
            s_fp32 = "(없음)"
        if b:
            x, y, w, h, cid, conf = b
            name = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else f"c{cid}"
            s_w8 = f"{name} {conf*100:.0f}% ({x},{y},{w},{h})"
        else:
            s_w8 = "(없음)"
        match = "O" if a == b else "X"
        print(f"  {i+1:2d}  {s_fp32:<45}  {s_w8:<45}  {match}")
    print()

    # 로그 파일이 있으면 L0 / total 요약만 출력
    ref_log = out_dir / "ref_fp32_log.txt"
    w8_log = out_dir / "w8_log.txt"
    if ref_log.exists():
        with open(ref_log) as f:
            for line in f:
                if "L0 " in line or "total=" in line:
                    print(f"  FP32 log: {line.rstrip()}")
    if w8_log.exists():
        with open(w8_log) as f:
            for line in f:
                if "L0 " in line or "total=" in line:
                    print(f"  W8 log:   {line.rstrip()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
