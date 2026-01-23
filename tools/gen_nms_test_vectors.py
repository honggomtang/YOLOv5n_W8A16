"""
NMS 테스트 벡터 생성
Python YOLOv5n의 NMS 결과와 비교하기 위한 참조 데이터 생성
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="assets/yolov5n.pt")
    ap.add_argument("--img", default="data/image/zidane.jpg")
    ap.add_argument("--out", default="tests/test_vectors_nms.h")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    ap.add_argument("--max-det", type=int, default=300, help="max detections before NMS")
    args = ap.parse_args()

    # 모델 로드
    model = YOLO(args.pt)
    
    # 이미지 로드 및 추론
    results = model(str(args.img), conf=args.conf, iou=args.iou, max_det=args.max_det, verbose=False)
    
    if len(results) == 0:
        print("No detections found")
        return 1
    
    result = results[0]
    
    # Detection 결과 추출
    boxes = result.boxes
    if len(boxes) == 0:
        print("No boxes found")
        return 1
    
    # boxes.data: [x1, y1, x2, y2, conf, cls]
    detections = boxes.data.cpu().numpy()
    
    # x1, y1, x2, y2를 center_x, center_y, width, height로 변환
    # 그리고 normalized (0-1)로 변환
    img_h, img_w = result.orig_shape
    detections_normalized = []
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        # 원본 이미지 좌표를 normalized로 변환
        center_x = ((x1 + x2) / 2.0) / img_w
        center_y = ((y1 + y2) / 2.0) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        detections_normalized.append([center_x, center_y, width, height, conf, int(cls)])
    
    detections_normalized = np.array(detections_normalized)
    
    print(f"Found {len(detections_normalized)} detections after NMS")
    print(f"Image size: {img_w}x{img_h}")
    
    # C 헤더 파일로 저장
    out_path = Path(args.out).expanduser().resolve()
    with out_path.open("w") as f:
        f.write("#ifndef TEST_VECTORS_NMS_H\n")
        f.write("#define TEST_VECTORS_NMS_H\n\n")
        f.write("// 자동 생성됨 (NMS 검증)\n")
        f.write("// Python YOLOv5n의 NMS 결과 (torchvision.ops.nms)\n\n")
        f.write('#include "../csrc/blocks/nms.h"\n\n')
        f.write(f"#define TV_NMS_NUM_DETECTIONS {len(detections_normalized)}\n")
        f.write(f"#define TV_NMS_IOU_THRESHOLD {args.iou}f\n")
        f.write(f"#define TV_NMS_MAX_DETECTIONS {args.max_det}\n\n")
        f.write("// Python NMS 결과 (confidence 내림차순 정렬됨)\n")
        f.write("static const detection_t tv_nms_detections[] = {\n")
        
        for det in detections_normalized:
            cx, cy, w, h, conf, cls = det
            f.write(f"    {{ {cx:.8e}f, {cy:.8e}f, {w:.8e}f, {h:.8e}f, {conf:.8e}f, {cls} }},\n")
        
        f.write("};\n\n")
        f.write("#endif // TEST_VECTORS_NMS_H\n")
    
    print(f"Wrote {len(detections_normalized)} detections to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
