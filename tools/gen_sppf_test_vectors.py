"""\
SPPF (Layer 9) 테스트 벡터 생성
입력 x랑 sppf 출력 y를 C 헤더로 뽑아줌
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="assets/yolov5n.pt")
    ap.add_argument("--out", default="tests/test_vectors_sppf.h")
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--w", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    from ultralytics import YOLO

    rng = np.random.default_rng(args.seed)
    # SPPF 입력: (1,256,H,W)
    x = rng.random((1, 256, args.h, args.w), dtype=np.float32)

    model = YOLO(args.pt).model
    layer9 = model.model[9]

    with torch.no_grad():
        xt = torch.from_numpy(x)
        yt = layer9(xt)
        y = yt.cpu().numpy().astype(np.float32)

    out_path = Path(args.out).expanduser().resolve()

    def dump_array(f, name: str, arr: np.ndarray) -> None:
        flat = arr.reshape(-1)
        f.write(f"static const float {name}[{flat.size}] = {{\n")
        for i, v in enumerate(flat.tolist()):
            f.write(f"{v:.8e}f,")
            if (i + 1) % 8 == 0:
                f.write("\n")
            else:
                f.write(" ")
        f.write("\n};\n\n")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("#ifndef TEST_VECTORS_SPPF_H\n")
        f.write("#define TEST_VECTORS_SPPF_H\n\n")
        f.write("// 자동 생성됨 (SPPF - Layer 9)\n\n")
        f.write(f"#define TV_SPPF_X_N 1\n#define TV_SPPF_X_C 256\n#define TV_SPPF_X_H {args.h}\n#define TV_SPPF_X_W {args.w}\n")
        f.write(f"#define TV_SPPF_Y_C {y.shape[1]}\n#define TV_SPPF_Y_H {y.shape[2]}\n#define TV_SPPF_Y_W {y.shape[3]}\n\n")
        dump_array(f, "tv_sppf_x", x)
        dump_array(f, "tv_sppf_y", y)
        f.write("#endif // TEST_VECTORS_SPPF_H\n")

    print(f"Wrote: {out_path}")
    print(f"입력 shape: {x.shape}, 출력 shape: {y.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
