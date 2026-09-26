"""Compare long autoregressive rollouts under different sampling settings. No training involved.

Replays the recorded keys of one episode through the model with each setting and writes a
side-by-side video: real game | setting 1 | setting 2 | ...

python compare_rollout.py --ckpt runs/wm/last.pt --episode data/raw/<file>.npz --frames 600
python compare_rollout.py --ckpt runs/wm/last.pt --episode data/raw/<file>.npz \
    --settings 3:0 10:0 10:0.1 20:0.1 --out compare.mp4

A setting is "steps:ctx_sigma" (denoising steps per frame : noise added to the context frames).
"""
import argparse
import time

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from wm.model import load_checkpoint
from wm.rollout import rollout


def label(frames, text):
    out = []
    for f in frames:
        img = Image.fromarray(f)
        d = ImageDraw.Draw(img)
        d.rectangle([0, f.shape[0] - 14, f.shape[1], f.shape[0]], fill=(0, 0, 0))
        d.text((3, f.shape[0] - 13), text, fill=(255, 230, 120))
        out.append(np.asarray(img))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episode", required=True, help="episode .npz (from data/raw)")
    ap.add_argument("--start", type=int, default=0, help="first context frame")
    ap.add_argument("--frames", type=int, default=600, help="frames to generate (15 per second)")
    ap.add_argument("--settings", nargs="+", default=["3:0", "10:0", "10:0.1"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default="compare.mp4")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_checkpoint(args.ckpt, dev)
    n = model.cfg.context
    d = np.load(args.episode)
    end = min(len(d["actions"]), args.start + n + args.frames)
    frames, actions = d["frames"][args.start:end], d["actions"][args.start:end]
    print(f"step {ck['step']} checkpoint, generating {len(frames) - n} frames per setting on {dev}")

    columns = [label(frames[n:], "real game")]
    for s in args.settings:
        steps, ctx_sigma = int(s.split(":")[0]), float(s.split(":")[1])
        torch.manual_seed(args.seed)
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.fp16 and dev == "cuda"):
            pred = rollout(model, frames, actions, steps=steps, ctx_sigma=ctx_sigma)
        print(f"  steps={steps} ctx_sigma={ctx_sigma}: {time.time() - t0:.0f}s")
        columns.append(label(pred, f"model steps={steps} ctx={ctx_sigma}"))

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    for t in range(len(columns[0])):
        writer.append_data(np.concatenate([c[t] for c in columns], 1))
    writer.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
