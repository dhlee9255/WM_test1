"""Measure generation speed for each sampling setting, to find what makes play_model slow.

python tools/bench_infer.py --ckpt checkpoints/last.pt
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wm_server"))
from wm.model import load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}, device: {torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU (no CUDA!)'}")
    if dev == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"GPU memory free {free / 1e9:.1f} / {total / 1e9:.1f} GB")
    model, ck = load_checkpoint(args.ckpt, dev)
    n, a = model.cfg.context, model.cfg.num_actions
    ctx = torch.zeros(1, n, 3, 240, 320, device=dev)
    acts = torch.zeros(1, n, a, device=dev)
    print(f"checkpoint step {ck['step']}\n")
    print(f"{'steps':>5} {'ctx_sigma':>9} {'fp16':>5} {'ms/frame':>9} {'fps':>6}")
    for steps in (3, 10):
        for fp16 in (False, True):
            for cs in (0.0, 0.1):
                if fp16 and dev != "cuda":
                    continue
                with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                    model.sample(ctx, acts, steps=steps, ctx_sigma=cs)  # warm-up
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    t = time.time()
                    for _ in range(args.reps):
                        model.sample(ctx, acts, steps=steps, ctx_sigma=cs)
                    if dev == "cuda":
                        torch.cuda.synchronize()
                dt = (time.time() - t) / args.reps
                print(f"{steps:>5} {cs:>9} {str(fp16):>5} {dt * 1000:>9.0f} {1 / dt:>6.1f}", flush=True)


if __name__ == "__main__":
    main()
