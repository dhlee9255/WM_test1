"""Shared game session (window or offscreen) and episode recorder."""
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from panda3d.core import loadPrcFileData

from . import core


class Session:
    """Frames come out at width x height; rendered at `supersample`x and box-filtered (anti-aliasing)."""

    def __init__(self, width, height, offscreen, title="WorldModel Game", supersample=2):
        self.ss = supersample
        loadPrcFileData("", "\n".join([
            f"win-size {width * supersample} {height * supersample}",
            f"window-type {'offscreen' if offscreen else 'onscreen'}",
            f"window-title {title}",
            "audio-library-name null",
            "sync-video 0",
            "notify-level-display error",
            "notify-level-glgsg error",
        ]))
        from direct.showbase.ShowBase import ShowBase
        from .render import Renderer
        self.base = ShowBase()
        self.renderer = Renderer(self.base)
        self.game = None

    def reset(self, seed):
        self.game = core.Game(seed)
        self.renderer.build(self.game)
        return self._render()

    def step(self, keys):
        info = self.game.step(keys)
        return self._render(), info

    def _render(self):
        self.renderer.sync(self.game)
        self.base.taskMgr.step()
        f = self.renderer.grab()
        if self.ss == 1:
            return f
        h, w = f.shape[0] // self.ss, f.shape[1] // self.ss
        f = f.reshape(h, self.ss, w, self.ss, 3).astype(np.uint16).sum((1, 3))
        return ((f + self.ss * self.ss // 2) // (self.ss * self.ss)).astype(np.uint8)


class Recorder:
    """frames[t] --actions[t]--> frames[t+1]. Saved as one .npz per episode."""

    def __init__(self, out_dir, width, height):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.size = (width, height)
        (self.out / "meta.json").write_text(json.dumps({
            "keys": core.KEYS, "tick_hz": core.TICK_HZ, "width": width, "height": height,
            "layout": "frames[t] --actions[t]--> frames[t+1]",
        }, indent=2))

    def start(self, frame, game, source):
        self.seed, self.source = game.seed, source
        self.frames, self.hp = [self._resize(frame)], [game.player.hp]
        self.actions, self.kills, self.damage, self.heal = [], [], [], []

    def add(self, keys, frame, info, game):
        self.actions.append(keys)
        self.frames.append(self._resize(frame))
        self.hp.append(game.player.hp)
        self.kills.append(info["kills"])
        self.damage.append(info["damage"])
        self.heal.append(info["heal"])

    def save(self, done):
        if not self.actions:
            return None
        path = self.out / f"{self.source}_{int(time.time() * 1000)}_s{self.seed}.npz"
        kills, damage = np.array(self.kills, np.int8), np.array(self.damage, np.int8)
        tmp = path.with_suffix(".tmp")  # write then rename, so a killed process never leaves a broken .npz
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                frames=np.stack(self.frames).astype(np.uint8),
                actions=np.array(self.actions, np.uint8),
                reward=(kills - 0.2 * damage).astype(np.float32),
                kills=kills, damage=damage, heal=np.array(self.heal, np.int8),
                hp=np.array(self.hp, np.int8),
                done=np.array(done), seed=np.array(self.seed),
            )
        os.replace(tmp, path)
        return path

    def _resize(self, frame):
        if (frame.shape[1], frame.shape[0]) == self.size:
            return frame
        return np.asarray(Image.fromarray(frame).resize(self.size, Image.BOX))
