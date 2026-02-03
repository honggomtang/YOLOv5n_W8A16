# -*- coding: utf-8 -*-
"""
weights.bin (FP32) → 레이어별 Symmetric Quantization → INT8 가중치 + scales.bin

- .weight 텐서만 INT8 양자화: scale = max(|w|) / 127, w_int8 = round(w/scale), clamp [-127,127].
- .bias 등 나머지는 FP32 유지.
- 출력: weights_w8.bin (메타데이터 + dtype별 데이터), scales.bin (INT8 텐서 순서대로 scale).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

# dtype: 0 = float32, 1 = int8 (C 로더와 약속)
DTYPE_FLOAT32 = 0
DTYPE_INT8 = 1

# symmetric int8 range (대칭 양자화)
INT8_MAX = 127
INT8_MIN = -127


def read_tensors(path: Path):
    """weights.bin (현재 FP32 포맷)을 파싱해 (key, shape, float32 bytes) 리스트 반환."""
    data = path.read_bytes()
    pos = 0
    end = len(data)
    if pos + 4 > end:
        raise ValueError("File too short")
    num_tensors = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    tensors = []
    for i in range(num_tensors):
        if pos + 4 > end:
            raise ValueError(f"Tensor {i}: truncated key_len")
        key_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if key_len > 1024 or pos + key_len > end:
            raise ValueError(f"Tensor {i}: invalid key_len or truncated key")
        key = data[pos : pos + key_len].decode("utf-8", errors="replace")
        pos += key_len

        if pos + 4 > end:
            raise ValueError(f"Tensor {i}: truncated ndim")
        ndim = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if ndim > 16 or pos + ndim * 4 > end:
            raise ValueError(f"Tensor {i}: invalid ndim or truncated shape")
        shape = list(struct.unpack_from("<" + "I" * ndim, data, pos))
        pos += ndim * 4

        # 4-byte align (C 로더와 동일)
        u = pos
        if u % 4 != 0:
            u = (u + 3) & ~3
            pos = u

        num_elems = 1
        for d in shape:
            num_elems *= d
        data_bytes = num_elems * 4
        if pos + data_bytes > end:
            raise ValueError(f"Tensor {i} ({key}): truncated data (need {data_bytes})")
        blob = bytes(data[pos : pos + data_bytes])
        pos += data_bytes
        tensors.append((key, shape, blob))
    return tensors


def symmetric_quantize_weight(w_blob: bytes, eps: float = 1e-8):
    """
    Symmetric quantization: scale = max(|w|) / 127, q = round(w/scale), clamp to [-127, 127].
    Returns (w_int8_bytes, scale).
    """
    n = len(w_blob) // 4
    max_abs = 0.0
    for i in range(n):
        x = struct.unpack_from("<f", w_blob, i * 4)[0]
        a = abs(x)
        if a > max_abs:
            max_abs = a
    scale = max_abs / INT8_MAX
    if scale < eps:
        scale = eps
    out = bytearray(n)
    for i in range(n):
        x = struct.unpack_from("<f", w_blob, i * 4)[0]
        q = round(x / scale)
        if q > INT8_MAX:
            q = INT8_MAX
        elif q < INT8_MIN:
            q = INT8_MIN
        out[i] = q & 0xFF
    return bytes(out), scale


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Symmetric quantize weights.bin (FP32) to INT8 per layer; output weights_w8.bin + scales.bin"
    )
    ap.add_argument("--weights", default="assets/weights.bin", help="입력 weights.bin (FP32)")
    ap.add_argument("--out-weights", default="assets/weights_w8.bin", help="출력 INT8/FP32 혼합 가중치")
    ap.add_argument("--out-scales", default=None, help="(선택) scales.bin 출력. 비우면 scale은 w8 내부에만 포함")
    ap.add_argument("--quiet", action="store_true", help="요약만 출력")
    args = ap.parse_args()

    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.exists():
        print(f"Error: Not found {weights_path}", file=sys.stderr)
        return 1

    tensors = read_tensors(weights_path)
    if not args.quiet:
        print(f"Read {len(tensors)} tensors from {weights_path}")

    scales_list = []
    out_weights_path = Path(args.out_weights).expanduser().resolve()
    out_scales_path = Path(args.out_scales).expanduser().resolve() if args.out_scales else None

    with out_weights_path.open("wb") as fw:
        fw.write(struct.pack("I", len(tensors)))

        for key, shape, blob in tensors:
            key_bytes = key.encode("utf-8")
            fw.write(struct.pack("I", len(key_bytes)))
            fw.write(key_bytes)
            fw.write(struct.pack("I", len(shape)))
            for d in shape:
                fw.write(struct.pack("I", d))

            if key.endswith(".weight"):
                w_int8_bytes, scale = symmetric_quantize_weight(blob)
                scales_list.append(scale)
                fw.write(struct.pack("B", DTYPE_INT8))
                fw.write(struct.pack("f", scale))  # A: scale을 w8 내부에 포함 (텐서별 매칭, 순서 독립)
                # 4B 정렬: dtype(1) + scale(4) = 5 → 패딩 3
                pos = fw.tell()
                pad = (4 - (pos % 4)) % 4
                if pad:
                    fw.write(b"\x00" * pad)
                fw.write(w_int8_bytes)
                if not args.quiet:
                    print(f"  [INT8] {key} shape={tuple(shape)} scale={scale:.6e}")
            else:
                fw.write(struct.pack("B", DTYPE_FLOAT32))
                pos = fw.tell()
                pad = (4 - (pos % 4)) % 4
                if pad:
                    fw.write(b"\x00" * pad)
                fw.write(blob)
                if not args.quiet:
                    print(f"  [FP32] {key} shape={tuple(shape)}")

        if out_scales_path:
            with out_scales_path.open("wb") as fs:
                fs.write(struct.pack("I", len(scales_list)))
                for s in scales_list:
                    fs.write(struct.pack("f", s))
            print(f"Wrote {out_scales_path} ({4 + len(scales_list)*4} B, {len(scales_list)} scales)")

    size_w8 = out_weights_path.stat().st_size
    size_orig = weights_path.stat().st_size
    print(f"Wrote {out_weights_path} ({size_w8 / (1024*1024):.2f} MB, scale per-tensor in w8)")
    print(f"Original weights.bin: {size_orig / (1024*1024):.2f} MB → W8 ~{100*size_w8/size_orig:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
