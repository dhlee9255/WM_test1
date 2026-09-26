from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedData:
    def __init__(self, packed_dir):
        d = Path(packed_dir)
        self.frames = np.load(d / "frames.npy", mmap_mode="r")
        self.actions = np.load(d / "actions.npy")
        self.episodes = np.load(d / "episodes.npy")

    def split(self, val):
        return [(int(s), int(n)) for s, n, v in self.episodes if bool(v) == val]


class WindowDataset(Dataset):
    """Item: (N+1) consecutive frames (uint8, N+1,H,W,3) and the N actions at the context frames."""

    def __init__(self, data: PackedData, context, val=False):
        self.data, self.n = data, context
        idx = [np.arange(s + context - 1, s + n - 1) for s, n in data.split(val) if n > context]
        self.index = np.concatenate(idx) if idx else np.zeros(0, np.int64)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = int(self.index[i])
        frames = torch.from_numpy(np.array(self.data.frames[j - self.n + 1:j + 2]))
        actions = torch.from_numpy(self.data.actions[j - self.n + 1:j + 1])
        return frames, actions


def to_model(frames_u8):
    """(B,T,H,W,3) uint8 -> (B,T,3,H,W) float in [-1,1] (on whatever device the input is)."""
    return frames_u8.permute(0, 1, 4, 2, 3).float().div(127.5).sub(1)


def to_image(x):
    """(..., 3, H, W) in [-1,1] -> (..., H, W, 3) uint8 numpy."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().movedim(-3, -1).cpu().numpy()
