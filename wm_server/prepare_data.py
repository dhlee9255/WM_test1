"""Pack episode .npz files (from collect.py / play.py --record) into flat arrays for training.

python prepare_data.py --raw data/raw --out data/packed

Writes:
  frames.npy    uint8 (total_frames, H, W, 3)   memory-mapped during training
  actions.npy   uint8 (total_frames, A)         action taken AT that frame (last frame of an episode: zeros)
  episodes.npy  int64 (E, 3)                    start, length, is_val
  meta.json
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/packed")
    ap.add_argument("--val-frac", type=float, default=0.03)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(raw.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files in {raw}")

    lengths = []
    for f in files:
        with np.load(f) as d:
            lengths.append(len(d["actions"]) + 1)
    with np.load(files[0]) as d:
        _, h, w, _ = d["frames"].shape
        num_actions = d["actions"].shape[1]
    total = sum(lengths)
    print(f"{len(files)} episodes, {total} frames, {h}x{w}, {total * h * w * 3 / 1e9:.1f} GB uncompressed")

    frames = np.lib.format.open_memmap(out / "frames.npy", mode="w+", dtype=np.uint8, shape=(total, h, w, 3))
    actions = np.zeros((total, num_actions), np.uint8)
    episodes = np.zeros((len(files), 3), np.int64)
    off = 0
    for i, (f, n) in enumerate(zip(files, lengths)):
        with np.load(f) as d:
            frames[off:off + n] = d["frames"]
            actions[off:off + n - 1] = d["actions"]
        # deterministic split by file name hash
        is_val = (zlib.crc32(f.name.encode()) % 10000) < args.val_frac * 10000
        episodes[i] = (off, n, int(is_val))
        off += n
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"  packed {i + 1}/{len(files)}", flush=True)
    frames.flush()
    np.save(out / "actions.npy", actions)
    np.save(out / "episodes.npy", episodes)
    meta = {"height": h, "width": w, "num_actions": num_actions, "frames": total,
            "episodes": len(files), "val_episodes": int(episodes[:, 2].sum())}
    src_meta = raw / "meta.json"
    if src_meta.exists():
        meta["source"] = json.loads(src_meta.read_text())
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done:", meta)


if __name__ == "__main__":
    main()
