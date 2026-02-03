#!/usr/bin/env python3
"""Detection 바이너리 디코딩 및 시각화."""

import argparse
import struct
from pathlib import Path
from dataclasses import dataclass

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMG_PATH = PROJECT_ROOT / "data" / "image" / "zidane.jpg"
DEFAULT_C_BIN_PATH = PROJECT_ROOT / "data" / "output" / "detections.bin"
DEFAULT_REF_BIN_PATH = PROJECT_ROOT / "data" / "output" / "ref" / "detections.bin"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_REF_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "ref"

DET_RECORD_SIZE = 12  # bytes: <HHHHBBBB

@dataclass
class Detection:
    x: int      # 중심 x (픽셀)
    y: int      # 중심 y (픽셀)
    w: int      # 너비 (픽셀)
    h: int      # 높이 (픽셀)
    class_id: int
    confidence: float  # 0.0 ~ 1.0
    
    @property
    def class_name(self):
        if 0 <= self.class_id < len(COCO_CLASSES):
            return COCO_CLASSES[self.class_id]
        return f"class_{self.class_id}"
    
    @property
    def x1(self):
        return self.x - self.w // 2
    
    @property
    def y1(self):
        return self.y - self.h // 2
    
    @property
    def x2(self):
        return self.x + self.w // 2
    
    @property
    def y2(self):
        return self.y + self.h // 2


def read_detections_bin(path: Path) -> list[Detection]:
    """바이너리 파일에서 detection 읽기"""
    if not path.exists():
        print(f"Error: {path} not found")
        return []
    
    detections = []
    with open(path, 'rb') as f:
        # 1 byte: detection 개수
        count = struct.unpack('B', f.read(1))[0]
        
        # 각 detection: 12 bytes (uint16 x,y,w,h + uint8 cls,conf,reserved[2])
        for _ in range(count):
            data = f.read(DET_RECORD_SIZE)
            if len(data) < DET_RECORD_SIZE:
                break
            x, y, w, h, cls_id, conf, r1, r2 = struct.unpack('<HHHHBBBB', data)
            detections.append(Detection(
                x=x, y=y, w=w, h=h,
                class_id=cls_id,
                confidence=conf / 255.0
            ))
    
    return detections


def write_detections_txt(detections: list[Detection], path: Path, title: str = ""):
    """텍스트 파일로 저장"""
    with open(path, 'w') as f:
        f.write(f"# {title}\n")
        f.write(f"# Detections: {len(detections)}\n")
        f.write("# Format: class_id class_name confidence x y w h\n\n")
        for d in detections:
            f.write(f"{d.class_id} {d.class_name:15s} {d.confidence:.4f} "
                   f"{d.x:4d} {d.y:4d} {d.w:4d} {d.h:4d}\n")
    print(f"Saved: {path}")


def visualize(detections: list[Detection], img_path: Path, out_path: Path, title: str = ""):
    """이미지에 bbox 그리기"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("PIL not available, skipping visualization")
        return
    
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return
    
    # 원본 이미지 로드 및 letterbox 적용 (C 코드와 동일)
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size
    
    # letterbox resize to 640x640
    scale = min(640 / orig_w, 640 / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    canvas = Image.new('RGB', (640, 640), (114, 114, 114))
    paste_x = (640 - new_w) // 2
    paste_y = (640 - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    
    draw = ImageDraw.Draw(canvas)
    
    # 색상 팔레트
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
    ]
    
    # 폰트 (시스템 기본 사용)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()
    
    # bbox 그리기
    for d in detections:
        color = colors[d.class_id % len(colors)]
        
        # bbox
        draw.rectangle([d.x1, d.y1, d.x2, d.y2], outline=color, width=2)
        
        # 라벨
        label = f"{d.class_name} {d.confidence:.2f}"
        bbox = draw.textbbox((d.x1, d.y1 - 20), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((d.x1, d.y1 - 20), label, fill=(255, 255, 255), font=font)
    
    # 타이틀
    if title:
        draw.text((10, 10), title, fill=(255, 255, 255), font=font)
    
    canvas.save(out_path)
    print(f"Saved: {out_path}")


def compare_detections(c_dets: list[Detection], ref_dets: list[Detection]):
    """C와 Python 결과 비교"""
    print("\n=== Comparison ===")
    print(f"C detections:   {len(c_dets)}")
    print(f"Ref detections: {len(ref_dets)}")
    
    if len(c_dets) != len(ref_dets):
        print("WARNING: Detection count mismatch!")
    
    # 클래스별 비교
    c_classes = sorted(set(d.class_id for d in c_dets))
    ref_classes = sorted(set(d.class_id for d in ref_dets))
    
    print(f"\nC classes:   {[COCO_CLASSES[c] if c < 80 else c for c in c_classes]}")
    print(f"Ref classes: {[COCO_CLASSES[c] if c < 80 else c for c in ref_classes]}")
    
    # 상위 detection 비교
    print("\nTop detections comparison:")
    print(f"{'C Result':<40} | {'Python Reference':<40}")
    print("-" * 85)
    
    max_show = max(len(c_dets), len(ref_dets))
    for i in range(min(5, max_show)):
        c_str = ""
        ref_str = ""
        if i < len(c_dets):
            d = c_dets[i]
            c_str = f"{d.class_name} {d.confidence:.3f} ({d.x},{d.y})"
        if i < len(ref_dets):
            d = ref_dets[i]
            ref_str = f"{d.class_name} {d.confidence:.3f} ({d.x},{d.y})"
        print(f"{c_str:<40} | {ref_str:<40}")


def main():
    parser = argparse.ArgumentParser(description="Decode and visualize detections (HW binary format)")
    parser.add_argument("--ref", action="store_true", help="Process reference (Python) result")
    parser.add_argument("--compare", action="store_true", help="Compare C and Python results")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--img", type=Path, default=DEFAULT_IMG_PATH, help="input image path (for visualization)")
    parser.add_argument("--c-bin", type=Path, default=DEFAULT_C_BIN_PATH, help="C detections.bin path")
    parser.add_argument("--ref-bin", type=Path, default=DEFAULT_REF_BIN_PATH, help="Python ref detections.bin path")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="output directory for C results")
    parser.add_argument("--ref-out-dir", type=Path, default=DEFAULT_REF_OUTPUT_DIR, help="output directory for ref results")
    parser.add_argument("--out-name", type=str, default="detections", help="output base name (e.g. detections → detections.txt, detections.jpg)")
    args = parser.parse_args()
    
    out_dir = args.out_dir.expanduser().resolve()
    ref_out_dir = args.ref_out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_txt = out_dir / f"{args.out_name}.txt"
    out_jpg = out_dir / f"{args.out_name}.jpg"

    if args.compare:
        # 둘 다 처리하고 비교
        print("=== C Result ===")
        c_dets = read_detections_bin(args.c_bin)
        if c_dets:
            write_detections_txt(c_dets, out_dir / "detections.txt", "C Detection Results")
            if not args.no_viz:
                visualize(c_dets, args.img, out_dir / "detections.jpg", "C Result")
        
        print("\n=== Python Reference ===")
        ref_dets = read_detections_bin(args.ref_bin)
        if ref_dets:
            ref_out_dir.mkdir(parents=True, exist_ok=True)
            write_detections_txt(ref_dets, ref_out_dir / "detections.txt", "Python Reference")
            if not args.no_viz:
                visualize(ref_dets, args.img, ref_out_dir / "detections.jpg", "Python Reference")
        
        if c_dets and ref_dets:
            compare_detections(c_dets, ref_dets)
    
    elif args.ref:
        # Python 참조만
        print("=== Python Reference ===")
        dets = read_detections_bin(args.ref_bin)
        if dets:
            ref_out_dir.mkdir(parents=True, exist_ok=True)
            write_detections_txt(dets, ref_out_dir / "detections.txt", "Python Reference")
            if not args.no_viz:
                visualize(dets, args.img, ref_out_dir / "detections.jpg", "Python Reference")
            
            print(f"\nDetections: {len(dets)}")
            for d in dets[:5]:
                print(f"  {d.class_name}: {d.confidence:.3f} at ({d.x}, {d.y})")
    
    else:
        # C 결과만
        print("=== C Result ===")
        dets = read_detections_bin(args.c_bin)
        if dets:
            write_detections_txt(dets, out_txt, "C Detection Results")
            if not args.no_viz:
                visualize(dets, args.img, out_jpg, "C Result")
            
            print(f"\nDetections: {len(dets)}")
            for d in dets[:5]:
                print(f"  {d.class_name}: {d.confidence:.3f} at ({d.x}, {d.y})")


if __name__ == "__main__":
    main()
