"""Make a small inference-only weights file from a training checkpoint (EMA weights in float16).

python tools/export_weights.py runs/wm_v2/last.pt wm_v2.pt
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="training checkpoint (last.pt)")
    ap.add_argument("dst", help="output file, e.g. wm_v2.pt")
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    ema = {k: v.half() if v.is_floating_point() else v for k, v in ck["ema"].items()}
    torch.save({"ema": ema, "config": ck["config"], "step": ck["step"]}, args.dst)
    n = sum(v.numel() for v in ema.values())
    print(f"step {ck['step']}, {n / 1e6:.1f}M params -> {args.dst}")


if __name__ == "__main__":
    main()
