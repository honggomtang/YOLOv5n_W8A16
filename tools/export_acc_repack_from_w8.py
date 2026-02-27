# -*- coding: utf-8 -*-
"""weights_w8.bin의 Conv int8 가중치를 가속기 스트림 순서로만 재배열 → weights_acc_repack.bin.
   (Scale_W는 weights_w8.bin 그대로, C 쪽 layers_config.h와 동일. 값은 int8 그대로, 레이아웃만 kw+kh*K+ic*K*K.)
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

NUM_CLUSTERS = 8
NUM_PE = 32

DTYPE_FLOAT32 = 0
DTYPE_INT8 = 1


def repack_conv_int8_for_acc(w: bytes, oc: int, ic: int, kh: int, kw: int) -> bytes:
    """int8 [OC, IC, KH, KW] (C order) → 가속기 순서 rd_w_addr = kw + kh*K + ic*K*K (kw가 가장 빠르게 증가), 주소당 8워드."""
    n = oc * ic * kh * kw
    if len(w) < n:
        raise ValueError(f"blob len {len(w)} < oc*ic*kh*kw={n}")
    out = []
    for oc_block in range((oc + NUM_PE - 1) // NUM_PE):
        oc0 = oc_block * NUM_PE
        for ic_ in range(ic):
            for kh_ in range(kh):
                for kw_ in range(kw):
                    for c in range(NUM_CLUSTERS):
                        word = 0
                        for pe in range(4):
                            oc_idx = oc0 + c * 4 + pe
                            if oc_idx < oc:
                                # C order: [oc, ic, kh, kw] → offset = oc*(ic*kh*kw) + ic*(kh*kw) + kh*kw + kw
                                off = oc_idx * (ic * kh * kw) + ic_ * (kh * kw) + kh_ * kw + kw_
                                val = w[off] & 0xFF
                                word |= val << (pe * 8)
                        out.append(struct.pack("<I", word))
    return b"".join(out)


def read_tensors_w8_with_blobs(path: Path):
    """weights_w8.bin 파싱. (key, shape, dtype, scale, blob) 리스트. blob은 bytes."""
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
        blob = bytes(data[pos : pos + data_bytes])
        pos += data_bytes
        tensors.append((key, shape, dtype, scale, blob))
    return tensors


def main() -> int:
    project = Path(__file__).resolve().parent.parent
    w8_path = project / "assets" / "weights_w8.bin"
    out_path = project / "assets" / "weights_acc_repack.bin"
    ap = __import__("argparse").ArgumentParser(description="weights_w8.bin → weights_acc_repack.bin (int8 순서만 재배열)")
    ap.add_argument("--weights", default=None, help="weights_w8.bin 경로")
    ap.add_argument("--out", default=None, help="출력 weights_acc_repack.bin 경로")
    args = ap.parse_args()
    if args.weights:
        w8_path = Path(args.weights).resolve()
    if args.out:
        out_path = Path(args.out).resolve()

    if not w8_path.exists():
        print(f"ERROR: not found: {w8_path}")
        return 1

    tensors = read_tensors_w8_with_blobs(w8_path)
    acc_chunks = []
    for key, shape, dtype, scale, blob in tensors:
        if not key.endswith(".conv.weight") and not key.endswith(".weight"):
            continue
        if len(shape) != 4 or dtype != DTYPE_INT8:
            continue
        oc, ic, kh, kw = shape[0], shape[1], shape[2], shape[3]
        repacked = repack_conv_int8_for_acc(blob, oc, ic, kh, kw)
        acc_chunks.append((key, repacked))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("I", len(acc_chunks)))
        for key, blob in acc_chunks:
            kb = key.encode("utf-8")
            f.write(struct.pack("I", len(kb)))
            f.write(kb)
            f.write(struct.pack("I", len(blob)))
            f.write(blob)
    print(f"Wrote {len(acc_chunks)} acc-repacked conv weights (from int8) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
