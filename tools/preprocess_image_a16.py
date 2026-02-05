"""이미지 전처리 (letterbox + normalize) → Q6.10 int16_t .bin 저장.

두 가지 입력 모드:
  1) --img image.jpg  : 이미지 로드 후 전처리·Q6.10 변환 (numpy, Pillow 사용 권장)
  2) --from-float bin : 기존 float32 .bin 읽어서 Q6.10 int16 .bin으로 변환 (표준 라이브러리만)

의존성: --img 사용 시 numpy, Pillow 필요. 프로젝트 루트에서:
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
  .venv/bin/python tools/preprocess_image_a16.py --img data/image/zidane.jpg --out data/input/preprocessed_image_a16.bin

헤더는 기존과 동일(24바이트): original_w, original_h, scale, pad_x, pad_y, size.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

# Q6.10: 소수 10비트 → 스케일 2^10
Q6_10_SCALE = 1024
HEADER_SIZE = 24


def convert_float_bin_to_a16(float_bin_path: Path, out_path: Path, quiet: bool) -> int:
    """기존 float32 전처리 .bin을 읽어 Q6.10 int16 .bin으로 저장 (numpy/PIL 불필요)."""
    data = float_bin_path.read_bytes()
    if len(data) < HEADER_SIZE:
        print(f"Error: file too small: {float_bin_path}")
        return 1

    original_w, original_h = struct.unpack_from("II", data, 0)
    scale, = struct.unpack_from("f", data, 8)
    paste_x, paste_y, size = struct.unpack_from("III", data, 12)

    n = 3 * size * size
    expected_float_bytes = n * 4
    if len(data) < HEADER_SIZE + expected_float_bytes:
        print(f"Error: expected {expected_float_bytes} float bytes after header")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(data[:HEADER_SIZE])
        for i in range(n):
            v, = struct.unpack_from("<f", data, HEADER_SIZE + i * 4)
            q = int(round(v * Q6_10_SCALE))
            q = max(0, min(32767, q))
            f.write(struct.pack("<h", q))

    if not quiet:
        file_size_kb = out_path.stat().st_size / 1024
        print(f"Converted: {float_bin_path} -> {out_path}")
        print(f"Shape: (3, {size}, {size}) Q6.10 int16, file size: {file_size_kb:.2f} KB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Preprocess image for YOLOv5n (W8A16): output Q6.10 int16_t .bin"
    )
    ap.add_argument("--img", help="입력 이미지 경로 (PIL 사용)")
    ap.add_argument(
        "--from-float",
        metavar="BIN",
        help="기존 float32 전처리 .bin 경로 (이 파일을 Q6.10 .bin으로 변환)",
    )
    ap.add_argument("--out", required=True, help="출력 .bin 파일 경로 (int16_t NCHW)")
    ap.add_argument("--size", type=int, default=640, help="--img 사용 시 리사이즈 크기")
    ap.add_argument("--quiet", action="store_true", help="로그 출력 비활성화")
    args = ap.parse_args()

    out_path = Path(args.out).expanduser().resolve()

    if args.from_float:
        float_path = Path(args.from_float).expanduser().resolve()
        if not float_path.is_file():
            print(f"Error: not a file: {float_path}")
            return 1
        return convert_float_bin_to_a16(float_path, out_path, args.quiet)

    if not args.img:
        print("Error: specify either --img IMAGE or --from-float BIN")
        return 1

    try:
        from PIL import Image
    except ImportError:
        print(
            "Error: --img requires Pillow. Install: pip install -r requirements.txt  (or use --from-float)"
        )
        return 1

    # 이미지 로드 및 전처리 (기존 preprocess_image_to_bin.py와 동일)
    img = Image.open(args.img).convert("RGB")
    original_w, original_h = img.size

    scale = min(args.size / original_w, args.size / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    img_padded = Image.new("RGB", (args.size, args.size), (114, 114, 114))
    paste_x = (args.size - new_w) // 2
    paste_y = (args.size - new_h) // 2
    img_padded.paste(img_resized, (paste_x, paste_y))

    w = h = args.size

    try:
        import numpy as np
        img_np = np.array(img_padded, dtype=np.float32) / 255.0
        img_nchw = img_np.transpose(2, 0, 1)
        img_q610 = np.clip(
            np.round(img_nchw * Q6_10_SCALE).astype(np.int32), 0, 32767
        ).astype(np.int16)
        nchw_bytes = img_q610.astype("<i2").tobytes()
        use_numpy = True
    except ImportError:
        use_numpy = False
        nchw = []
        for c in range(3):
            for y in range(h):
                for x in range(w):
                    v = img_padded.getpixel((x, y))[c] / 255.0
                    q = max(0, min(32767, int(round(v * Q6_10_SCALE))))
                    nchw.append(q)

    if not args.quiet:
        print(f"Original size: {original_w}x{original_h}")
        print(f"Resized size: {new_w}x{new_h}")
        print(f"Padded size: {args.size}x{args.size}")
        print(f"Image shape: (3, {h}, {w}) (Q6.10 int16 NCHW)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("I", original_w))
        f.write(struct.pack("I", original_h))
        f.write(struct.pack("f", scale))
        f.write(struct.pack("I", paste_x))
        f.write(struct.pack("I", paste_y))
        f.write(struct.pack("I", args.size))
        if use_numpy:
            f.write(nchw_bytes)
        else:
            for v in nchw:
                f.write(struct.pack("<h", v))

    if not args.quiet:
        file_size_kb = out_path.stat().st_size / 1024
        print(f"\nWrote: {out_path}")
        print(f"File size: {file_size_kb:.2f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
