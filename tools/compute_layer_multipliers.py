#!/usr/bin/env python3
"""
각 레이어(Conv weight)의 Scale_W를 구하고 Multiplier = Scale_W * 65536 으로 계산해
layers_config.h용 C 매크로를 출력/갱신.

데이터 소스:
  - weights_w8.bin: 파일에 저장된 Scale_W 사용 (INT8 텐서별 scale)
  - weights.bin (FP32): --from-fp32 시 Symmetric Scale_W = max(|w|)/127 로 계산

사용:
  python3 tools/compute_layer_multipliers.py --from-fp32 --output csrc/layers_config.h
  python3 tools/compute_layer_multipliers.py --weights assets/weights_w8.bin --output csrc/layers_config.h
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

# weights_w8.bin: num_tensors(4), then per tensor: key_len(4), key, ndim(4), shape[], dtype(1)
# dtype 0=float32 → align 4 → float32 data
# dtype 1=int8   → scale(4) → align 4 → int8 data
DTYPE_FLOAT32 = 0
DTYPE_INT8 = 1


def read_tensors_fp32(path: Path):
    """weights.bin (FP32) 파싱: (key, shape, blob) 리스트."""
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
    """Symmetric quantization scale only: max(|w|)/127 (quantize_weights와 동일)."""
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


def read_tensors_w8(path: Path):
    """weights_w8.bin 파싱. (key, shape, dtype, scale) 리스트 반환. scale은 INT8일 때만 유효."""
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
        if pos + 1 > end:
            raise ValueError(f"Tensor {i}: truncated dtype")
        dtype = data[pos]
        pos += 1
        scale = None
        if dtype == DTYPE_INT8:
            if pos + 4 > end:
                raise ValueError(f"Tensor {i}: truncated scale")
            scale = struct.unpack_from("<f", data, pos)[0]
            pos += 4
        # 4-byte align
        pos = (pos + 3) & ~3
        num_elems = 1
        for d in shape:
            num_elems *= d
        if dtype == DTYPE_FLOAT32:
            data_bytes = num_elems * 4
        else:
            data_bytes = num_elems * 1
        if pos + data_bytes > end:
            raise ValueError(f"Tensor {i} ({key}): truncated data")
        pos += data_bytes
        tensors.append((key, shape, dtype, scale))
    return tensors


# 레이어 인덱스 0..24에 대응하는 대표 conv weight 키 (main.c W8A16 순서와 맞춤)
LAYER_WEIGHT_KEYS = [
    "model.0.conv.weight",       # L0
    "model.1.conv.weight",      # L1
    "model.2.cv1.conv.weight",   # L2
    "model.3.conv.weight",      # L3
    "model.4.cv1.conv.weight",  # L4
    "model.5.conv.weight",      # L5
    "model.6.cv1.conv.weight",  # L6
    "model.7.conv.weight",     # L7
    "model.8.cv1.conv.weight", # L8
    "model.9.cv1.conv.weight", # L9
    "model.10.conv.weight",    # L10
    "model.10.conv.weight",    # L11 upsample (reuse L10)
    "model.13.cv1.conv.weight", # L12 concat (use L13)
    "model.13.cv1.conv.weight", # L13
    "model.14.conv.weight",    # L14
    "model.14.conv.weight",    # L15 upsample
    "model.17.cv1.conv.weight", # L16 concat (use L17)
    "model.17.cv1.conv.weight", # L17
    "model.18.conv.weight",    # L18
    "model.18.conv.weight",    # L19 concat
    "model.20.cv1.conv.weight", # L20
    "model.21.conv.weight",    # L21
    "model.21.conv.weight",    # L22 concat
    "model.23.cv1.conv.weight", # L23
    "model.24.m.0.weight",     # L24 detect
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract Scale_W and compute Multiplier = Scale_W*65536 for layers_config.h"
    )
    ap.add_argument("--weights", default=None, help="weights_w8.bin 또는 weights.bin 경로")
    ap.add_argument("--from-fp32", action="store_true",
                    help="weights.bin(FP32)에서 Scale_W = max(|w|)/127 로 계산 (권장: 레이어별 실제 값)")
    ap.add_argument("--output", default=None, help="출력 헤더 경로 (비우면 stdout만)")
    args = ap.parse_args()

    if args.from_fp32:
        weights_path = Path(args.weights or "assets/weights.bin").resolve()
        if not weights_path.exists():
            print(f"Error: Not found {weights_path}", file=sys.stderr)
            return 1
        tensors_fp32 = read_tensors_fp32(weights_path)
        key_to_scale = {}
        for key, shape, blob in tensors_fp32:
            if key.endswith(".weight"):
                # PyTorch export 시 "model.model.model.0..." 형태일 수 있음 → "model.0..." 로 통일
                norm_key = key
                if key.startswith("model.model.model."):
                    norm_key = "model." + key[len("model.model.model."):]
                key_to_scale[norm_key] = symmetric_scale_only(blob)
        source_name = weights_path.name + " (FP32, Scale_W=max|w|/127)"
    else:
        weights_path = Path(args.weights or "assets/weights_w8.bin").resolve()
        if not weights_path.exists():
            print(f"Error: Not found {weights_path}", file=sys.stderr)
            return 1
        tensors = read_tensors_w8(weights_path)
        key_to_scale = {}
        for key, shape, dtype, scale in tensors:
            if dtype == DTYPE_INT8 and scale is not None and key.endswith(".weight"):
                key_to_scale[key] = scale
        source_name = weights_path.name

    # Multiplier = Scale_W * 65536 (round), uint32_t 범위로 클램프
    results = []
    for i, key in enumerate(LAYER_WEIGHT_KEYS):
        scale = key_to_scale.get(key)
        if scale is None or scale <= 0:
            # fallback: 1/1024 → 64
            scale = 1.0 / 1024.0
            mult = 64
            note = " (fallback)"
        else:
            raw = scale * 65536.0
            mult = int(round(raw))
            if mult < 1:
                mult = 1
            if mult > 0xFFFFFFFF:
                mult = 0xFFFFFFFF
            note = ""
        results.append((i, key, scale, mult, note))

    # 표 출력
    print("/* Multiplier = Scale_W * 65536 from", source_name, "*/")
    for i, key, scale, mult, note in results:
        print(f"  L{i}: {key}  Scale_W={scale:.6f}  -> LAYER_MULTIPLIER_{i}={mult}{note}")

    # 검증: 모두 64가 아니어야 함 (진짜 레이어별 값)
    all_64 = all(m == 64 for (_, _, _, m, _) in results)
    if all_64:
        print("\n  [WARN] All multipliers are 64. Check that weights_w8.bin has per-layer scales.", file=sys.stderr)
    else:
        unique = len(set(m for (_, _, _, m, _) in results))
        print(f"\n  [OK] Multipliers vary: {unique} distinct values (not all 64).")

    # C 매크로
    lines = [
        "/**",
        " * W8A16 Calibration Map: 레이어별 Fixed-point Multiplier",
        " *",
        " * Multiplier = Scale_W * 65536 (Q0.16). Requant: out = (acc * multiplier + 32768) >> 16",
        " * Data: " + source_name + " 에서 Scale_W 적용.",
        " *",
        " * Bias (동일): Bias_q = round(Bias_f * 1024 / Scale_W)",
        " */",
        "#ifndef LAYERS_CONFIG_H",
        "#define LAYERS_CONFIG_H",
        "",
        "#include <stdint.h>",
        "",
        "/* model.0 ~ model.24: multiplier = Scale_W * 65536 (from weights_w8.bin) */",
    ]
    for i, key, scale, mult, _ in results:
        lines.append(f"#define LAYER_MULTIPLIER_{i}   {mult}U")
    lines.append("")
    lines.append("#endif /* LAYERS_CONFIG_H */")
    lines.append("")

    out_content = "\n".join(lines)
    if args.output:
        out_path = Path(args.output).resolve()
        out_path.write_text(out_content, encoding="utf-8")
        print(f"\nWrote {out_path}")
    else:
        print("\n/* C macros for layers_config.h (copy or use --output csrc/layers_config.h) */")
        print(out_content)

    return 0


if __name__ == "__main__":
    sys.exit(main())
