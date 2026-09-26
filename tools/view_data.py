"""Render recorded episodes as a 2x2 video with key/HP overlays, to inspect training data.

python tools/view_data.py --seeds 5 150 20 passive --seconds 30 --speed 2 --out data_review.mp4
"""
import argparse
import glob
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

KEYS = ["W", "A", "S", "D", "ATK", "E"]
BAR = 44


def load_episode(raw, seed):
    f = [p for p in glob.glob(str(Path(raw) / "*.npz")) if re.search(rf"_s{seed}\.npz$", p)]
    if not f:
        raise SystemExit(f"no episode with seed {seed} in {raw}")
    d = np.load(f[0])
    return d["frames"], d["actions"], d["hp"], np.cumsum(np.r_[0, d["kills"]])


def passive_chain(raw, n_frames):
    """Concatenate stand-and-die episodes until n_frames."""
    fr, ac, hp, kl = [], [], [], []
    for f in sorted(glob.glob(str(Path(raw) / "passive_*.npz"))):
        d = np.load(f)
        fr.append(d["frames"][:-1]); ac.append(d["actions"]); hp.append(d["hp"][:-1]); kl.append(np.zeros(len(d["actions"]), int))
        if sum(len(a) for a in ac) >= n_frames:
            break
    return np.concatenate(fr), np.concatenate(ac), np.concatenate(hp), np.concatenate(kl)


def tile(frame, action, hp, kills, title, t, font):
    h, w, _ = frame.shape
    img = Image.new("RGB", (w, h + BAR), (18, 18, 22))
    img.paste(Image.fromarray(frame), (0, 0))
    d = ImageDraw.Draw(img)
    d.text((4, h + 2), f"{title}  t={t}  hp={hp}  kills={kills}", fill=(220, 220, 220), font=font)
    x = 4
    for k, on in zip(KEYS, action):
        bw = 14 + 7 * len(k)
        d.rectangle([x, h + 20, x + bw, h + 40], fill=(230, 180, 40) if on else (55, 55, 62))
        d.text((x + 5, h + 23), k, fill=(0, 0, 0) if on else (150, 150, 150), font=font)
        x += bw + 5
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="wm_server/data/raw")
    ap.add_argument("--seeds", nargs=4, default=["5", "150", "20", "passive"])
    ap.add_argument("--titles", nargs=4, default=["old wander bot", "hunting bot", "smooth bot", "stand & die"])
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--speed", type=float, default=2)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="data_review.mp4")
    args = ap.parse_args()

    fps, data_hz = 15 * args.speed, 15
    n = int(args.seconds * fps)
    eps = [passive_chain(args.raw, args.start + n) if s == "passive" else load_episode(args.raw, int(s)) for s in args.seeds]
    font = ImageFont.load_default()
    writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    for t in range(args.start, args.start + n):
        tiles = []
        for (fr, ac, hp, kl), title in zip(eps, args.titles):
            i = min(t, len(ac) - 1)
            tiles.append(tile(fr[i], ac[i], int(hp[i]), int(kl[i]), title, i, font))
        grid = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
        grid = np.asarray(Image.fromarray(grid).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.NEAREST))
        writer.append_data(grid)
    writer.close()
    print(f"wrote {args.out}: {n} frames at {fps:.0f} fps ({n / fps:.0f}s video = {n / data_hz:.0f}s of gameplay)")


if __name__ == "__main__":
    main()
