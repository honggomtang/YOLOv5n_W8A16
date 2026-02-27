# -*- coding: utf-8 -*-
"""PyTorch .pt → C weights.bin (Fused 모델 지원).
   옵션 --acc-repack: Conv 가중치를 가속기 레이아웃 [OC/32][IC][KH][KW][8클러스터][4채널] 스트림 순서로
   추가 출력 (weights_acc_repack.bin). 하드웨어: rd_w_addr = kw + kh*K + ic*K*K, 매 주소마다 8워드.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import struct
import numpy as np
import torch

NUM_CLUSTERS = 8
NUM_PE = 32


def repack_conv_weight_for_acc(
    w: np.ndarray,
    oc: int,
    ic: int,
    kh: int,
    kw: int,
) -> bytes:
    """[OC, IC, KH, KW] → 가속기 스트림 순서: rd_w_addr = kw + kh*K + ic*K*K (RTL과 동일).
    주소마다 8워드(클러스터 0~7), 워드당 4바이트(PE 4개 int8). OC는 32 단위로 패딩.
    """
    if w.dtype == np.float32:
        w = np.clip(np.round(w).astype(np.int32), -128, 127).astype(np.int8)
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
                                # w shape (OC, IC, KH, KW)
                                val = int(w[oc_idx, ic_, kh_, kw_]) & 0xFF
                                word |= val << (pe * 8)
                        out.append(struct.pack("<I", word))
    return b"".join(out)

_YOLOv5ModelStub = type("Model", (torch.nn.Module,), {})


class _YOLOv5Unpickler(pickle.Unpickler):
    """YOLOv5 .pt 로드용 스텁 (models.* 대체)."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "models.yolo" and name == "Model":
            return _YOLOv5ModelStub
        if module.startswith("models."):
            return type(name, (torch.nn.Module,), {})
        return super().find_class(module, name)


def _make_stub_pickle_module():
    m = types.ModuleType("_stub_pickle")
    m.Unpickler = _YOLOv5Unpickler
    for attr in ("load", "loads", "dump", "dumps", "PROTOCOL", "HIGHEST_PROTOCOL"):
        if hasattr(pickle, attr):
            setattr(m, attr, getattr(pickle, attr))
    return m


def _load_state_dict(obj: Any) -> Dict[str, Any]:
    """YOLOv5 .pt에서 state_dict 추출."""
    # Plain state_dict case
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj:
            m = obj["model"]
            if hasattr(m, "state_dict"):
                return m.state_dict()
            if isinstance(m, dict):
                # Sometimes nested state dict
                return m
        if "ema" in obj:
            m = obj["ema"]
            if hasattr(m, "state_dict"):
                return m.state_dict()
            if isinstance(m, dict):
                return m

        # Heuristic: treat as state_dict if many keys end with weight/bias/running_*
        suffix_hits = 0
        for k in obj.keys():
            if any(k.endswith(s) for s in (".weight", ".bias", ".running_mean", ".running_var")):
                suffix_hits += 1
        if suffix_hits >= 3:
            return obj

    # Full model object case
    if hasattr(obj, "state_dict"):
        return obj.state_dict()

    raise ValueError(
        "Could not locate a state_dict inside the .pt file. "
        "Try exporting a checkpoint that contains model parameters."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="assets/yolov5n.pt")
    ap.add_argument("--out", default="assets/weights.bin")
    ap.add_argument("--trust-pickle", action="store_true", help="PyTorch 2.6+ weights_only=False")
    ap.add_argument(
        "--classic",
        action="store_true",
        help="Anchor-based (Standard YOLOv5n). torch.hub ultralytics/yolov5 custom. "
        "Needs network. Use for detections_ref match (desktop detect.py).",
    )
    ap.add_argument(
        "--acc-repack",
        action="store_true",
        help="Conv 가중치를 가속기 스트림 레이아웃으로 추가 출력 (weights_acc_repack.bin).",
    )
    args = ap.parse_args()

    pt_path = Path(args.pt).expanduser().resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"PT file not found: {pt_path}")

    state_dict = None

    if args.classic:
        # Anchor-based YOLOv5n (121 tensors). torch.hub만 사용 — 스텁 폴백은 349개(.bias 누락)라 C 호환 불가.
        try:
            import ultralytics  # noqa: F401
        except ModuleNotFoundError:
            raise SystemExit(
                "ERROR: ultralytics not installed for this Python.\n"
                f"Current Python: {sys.executable}\n"
                "Run: py -3.11 -m pip install ultralytics torch numpy\n"
                "Then run this script again with: py -3.11 tools/export_weights_to_bin.py --pt ... --out ... --classic"
            )
        print(f"Using Python: {sys.executable}")
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(pt_path),
            force_reload=False,
            trust_repo=True,
        )
        state_dict = model.state_dict()
        print("Loaded model using torch.hub ultralytics/yolov5 (classic)")
        if len(state_dict) != 121:
            print(f"Note: state_dict has {len(state_dict)} tensors (expected 121 for YOLOv5n)")

    if state_dict is None:
        # DFL(Ultralytics) 또는 torch.load 폴백
        try:
            from ultralytics import YOLO
            model = YOLO(str(pt_path))
            state_dict = model.model.state_dict()
            print("Loaded model using ultralytics (DFL)")
        except Exception:
            try:
                ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=True)
                state_dict = _load_state_dict(ckpt)
            except Exception as e:
                if args.trust_pickle:
                    try:
                        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                        state_dict = _load_state_dict(ckpt)
                    except ModuleNotFoundError as mnfe:
                        raise ModuleNotFoundError(
                            f"Checkpoint requires YOLOv5 code to be importable.\n"
                            f"Options: 1) pip install ultralytics, 2) Run inside yolov5 repo.\n"
                            f"Error: {mnfe}"
                        ) from mnfe
                else:
                    raise RuntimeError(
                        f"Failed to load .pt with weights_only=True (PyTorch 2.6+).\n"
                        f"If trusted, retry with --trust-pickle.\n"
                        f"Error: {type(e).__name__}: {e}"
                    ) from e

    # 바이너리 파일로 저장
    out_path = Path(args.out).expanduser().resolve()
    
    acc_repack_chunks: list[Tuple[str, bytes]] = []
    
    with out_path.open("wb") as f:
        # 헤더: 텐서 개수 (4 bytes)
        num_tensors = len(state_dict)
        f.write(struct.pack("I", num_tensors))
        
        # 각 텐서 저장
        for key, tensor in state_dict.items():
            # 키 이름 (길이 + 문자열)
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("I", len(key_bytes)))
            f.write(key_bytes)
            
            # 텐서 shape (차원 수 + 각 차원 크기)
            shape = tensor.shape
            f.write(struct.pack("I", len(shape)))
            for dim in shape:
                f.write(struct.pack("I", dim))
            
            # RISC-V 등: float 데이터를 4바이트 정렬 (misalign trap 방지)
            pos = f.tell()
            pad = (4 - (pos % 4)) % 4
            if pad:
                f.write(b"\x00" * pad)
            
            # 텐서 데이터 (float32)
            data = tensor.cpu().numpy().astype(np.float32)
            f.write(data.tobytes())
            
            # 가속기 재배열: .conv.weight 이고 4D (OC, IC, KH, KW) 인 경우
            if args.acc_repack and key.endswith(".conv.weight") and len(shape) == 4:
                oc, ic, kh, kw = int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
                repacked = repack_conv_weight_for_acc(data, oc, ic, kh, kw)
                acc_repack_chunks.append((key, repacked))
    
    print(f"Wrote {num_tensors} tensors to {out_path}")
    print(f"File size: {out_path.stat().st_size / (1024*1024):.2f} MB")
    
    if args.acc_repack and acc_repack_chunks:
        acc_path = out_path.parent / (out_path.stem + "_acc_repack.bin")
        with acc_path.open("wb") as af:
            af.write(struct.pack("I", len(acc_repack_chunks)))
            for key, blob in acc_repack_chunks:
                kb = key.encode("utf-8")
                af.write(struct.pack("I", len(kb)))
                af.write(kb)
                af.write(struct.pack("I", len(blob)))
                af.write(blob)
        print(f"Wrote {len(acc_repack_chunks)} acc-repacked conv weights to {acc_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
