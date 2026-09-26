"""Compare long autoregressive rollouts under different sampling settings. No training involved.

    python compare_rollout.py

Picks a validation episode (not used for training), replays its recorded keys through the model
for 40 seconds with each setting, and writes compare.mp4: real game | setting 1 | setting 2 | ...
A setting is "steps:ctx_sigma" (denoising steps per frame : noise added to the context frames).
Options: --ckpt, --episode, --frames, --settings, --out (see --help).
"""
import argparse
import time
import zlib
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from wm.model import fast_fp16_available, load_checkpoint
from wm.rollout import rollout


def pick_episode(raw, min_len):
    """A validation episode by the same hash split as prepare_data.py (3%), long enough."""
    files = sorted(Path(raw).glob("bot_*.npz"))
    val = [f for f in files if (zlib.crc32(f.name.encode()) % 10000) < 300]
    for f in val + files:
        with np.load(f) as d:
            if len(d["actions"]) >= min_len:
                return f
    raise SystemExit(f"no episode with {min_len}+ frames in {raw}")


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
    ap.add_argument("--ckpt", default="runs/wm/last.pt")
    ap.add_argument("--episode", default=None, help="episode .npz (default: a validation episode from data/raw)")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--start", type=int, default=100, help="first context frame")
    ap.add_argument("--frames", type=int, default=600, help="frames to generate (15 per second)")
    ap.add_argument("--settings", nargs="+", default=["3:0.1", "5:0.1", "7:0.1", "10:0.1"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="compare.mp4")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    half = dev == "cuda" and fast_fp16_available()
    torch.backends.cudnn.benchmark = True
    model, ck = load_checkpoint(args.ckpt, dev, half=half)
    n = model.cfg.context
    episode = Path(args.episode) if args.episode else pick_episode(args.raw, args.start + n + args.frames)
    d = np.load(episode)
    end = min(len(d["actions"]), args.start + n + args.frames)
    frames, actions = d["frames"][args.start:end], d["actions"][args.start:end]
    print(f"checkpoint step {ck['step']}, episode {episode.name}, {len(frames) - n} frames per setting, "
          f"{dev}{' fp16' if half else ''}")

    columns = [label(frames[n:], "real game")]
    for s in args.settings:
        steps, ctx_sigma = int(s.split(":")[0]), float(s.split(":")[1])
        torch.manual_seed(args.seed)
        t0 = time.time()
        pred = rollout(model, frames, actions, steps=steps, ctx_sigma=ctx_sigma)
        print(f"  steps={steps} ctx_sigma={ctx_sigma}: {time.time() - t0:.0f}s", flush=True)
        columns.append(label(pred, f"steps={steps} ctx={ctx_sigma}"))

    writer = imageio.get_writer(args.out, fps=15, codec="libx264", quality=8, macro_block_size=8)
    for t in range(len(columns[0])):
        writer.append_data(np.concatenate([c[t] for c in columns], 1))
    writer.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
