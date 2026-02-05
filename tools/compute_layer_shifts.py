#!/usr/bin/env python3
"""
weights.bin (FP32)에서 레이어별 Scale_W를 구하고,
shift = round(log2(1.0 / Scale_W)) 로 LAYER_SHIFT 값을 계산.
출력: C 매크로 또는 표 (layers_config.h 갱신용).
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

# quantize_weights와 동일한 read_tensors
def read_tensors(path: Path):
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
            raise ValueError(f"Tensor {i}: invalid key_len")
        key = data[pos : pos + key_len].decode("utf-8", errors="replace")
        pos += key_len
        if pos + 4 > end:
            raise ValueError(f"Tensor {i}: truncated ndim")
        ndim = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if ndim > 16 or pos + ndim * 4 > end:
            raise ValueError(f"Tensor {i}: invalid ndim")
        shape = list(struct.unpack_from("<" + "I" * ndim, data, pos))
        pos += ndim * 4
        u = pos
        if u % 4 != 0:
            u = (u + 3) & ~3
            pos = u
        num_elems = 1
        for d in shape:
            num_elems *= d
        data_bytes = num_elems * 4
        if pos + data_bytes > end:
            raise ValueError(f"Tensor {i} ({key}): truncated data")
        blob = bytes(data[pos : pos + data_bytes])
        pos += data_bytes
        tensors.append((key, shape, blob))
    return tensors


def symmetric_scale_only(w_blob: bytes, eps: float = 1e-8) -> float:
    """Symmetric quantization scale only (max(|w|)/127)."""
    n = len(w_blob) // 4
    max_abs = 0.0
    for i in range(n):
        x = struct.unpack_from("<f", w_blob, i * 4)[0]
        a = abs(x)
        if a > max_abs:
            max_abs = a
    scale = max_abs / 127.0
    if scale < eps:
        scale = eps
    return scale


# 레이어 인덱스 0..24에 대응하는 대표 conv weight 키 (main.c W8A16 순서와 맞춤)
LAYER_WEIGHT_KEYS = [
    "model.0.conv.weight",      # L0
    "model.1.conv.weight",     # L1
    "model.2.cv1.conv.weight", # L2
    "model.3.conv.weight",     # L3
    "model.4.cv1.conv.weight", # L4
    "model.5.conv.weight",     # L5
    "model.6.cv1.conv.weight", # L6
    "model.7.conv.weight",     # L7
    "model.8.cv1.conv.weight", # L8
    "model.9.cv1.conv.weight", # L9
    "model.10.conv.weight",     # L10
    "model.10.conv.weight",    # L11 upsample (reuse L10)
    "model.13.cv1.conv.weight",# L12 concat (use L13)
    "model.13.cv1.conv.weight",# L13
    "model.14.conv.weight",     # L14
    "model.14.conv.weight",    # L15 upsample
    "model.17.cv1.conv.weight",# L16 concat (use L17)
    "model.17.cv1.conv.weight",# L17
    "model.18.conv.weight",     # L18
    "model.18.conv.weight",    # L19 concat
    "model.20.cv1.conv.weight",# L20
    "model.21.conv.weight",     # L21
    "model.21.conv.weight",    # L22 concat
    "model.23.cv1.conv.weight",# L23
    "model.24.m.0.weight",     # L24 detect
]


def main() -> int:
    weights_path = Path("assets/weights.bin").resolve()
    if not weights_path.exists():
        print(f"Error: Not found {weights_path}", file=sys.stderr)
        return 1

    tensors = read_tensors(weights_path)
    key_to_scale = {}
    for key, shape, blob in tensors:
        if not key.endswith(".weight"):
            continue
        scale = symmetric_scale_only(blob)
        key_to_scale[key] = scale

    # shift = round(log2(1.0 / Scale_W)); clamp to [0, 16]
    shifts = []
    for i, key in enumerate(LAYER_WEIGHT_KEYS):
        scale = key_to_scale.get(key)
        if scale is None or scale <= 0:
            # fallback
            shift = 10
            scale = 1.0 / (1 << 10)
        else:
            inv = 1.0 / scale
            shift = int(round(math.log2(inv)))
            if shift < 0:
                shift = 0
            if shift > 16:
                shift = 16
        shifts.append((i, key, scale, shift))

    print("/* shift = round(log2(1.0 / Scale_W)) from assets/weights.bin */")
    for i, key, scale, shift in shifts:
        print(f"  L{i}: {key}  Scale_W={scale:.6e}  -> shift={shift}")

    print("\n/* C macros for layers_config.h */")
    for i, _, _, shift in shifts:
        print(f"#define LAYER_SHIFT_{i}   {shift}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
