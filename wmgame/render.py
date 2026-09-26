"""Panda3D renderer that mirrors a core.Game state. Same code path for window and offscreen."""
import math
import random
from pathlib import Path

import numpy as np
import simplepbr
from direct.actor.Actor import Actor
from panda3d.core import (AmbientLight, CardMaker, DirectionalLight, Filename, GraphicsOutput,
                          NodePath, Texture, TransparencyAttrib)

from . import core

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "processed"
ANIM = {"idle": "idle", "walk": "walk", "attack": "attack-melee-right", "pickup": "pick-up", "die": "die"}
LOOPING = {"idle", "walk"}
PROP_MODEL = {core.WALL: "wall", core.ROCKS: "rocks", core.STONES: "stones", core.COLUMN: "column",
              core.BARREL: "barrel", core.TABLE: "table", core.CHAIR: "chair",
              core.CHEST: "chest", core.POT: "pot", core.WOOD: "wood-structure"}
ACTION_TICKS = {"attack": core.ATTACK_TICKS, "pickup": core.PICKUP_TICKS}
ENEMY_MODELS = {"skeleton": "graveyard/character-skeleton", "zombie": "graveyard/character-zombie"}
POOL_PER_KIND = 16
CAM_OFFSET = (0.0, -3.8, 4.4)


def _asset(name):
    return Filename.fromOsSpecific(str(ASSETS / f"{name}.glb"))


class Renderer:
    def __init__(self, base):
        if not (ASSETS / "character-human.glb").exists():
            raise FileNotFoundError("run `python setup_assets.py` first")
        self.base = base
        base.setBackgroundColor(0.08, 0.08, 0.1, 1)
        base.disableMouse()
        aspect = base.win.getXSize() / base.win.getYSize()
        vfov = 44
        hfov = math.degrees(2 * math.atan(math.tan(math.radians(vfov / 2)) * aspect))
        base.camLens.setFov(hfov, vfov)
        self._models = {}
        self._setup_lights()
        self._setup_hud()

        self.map_root = None
        self.player = self._make_character("character-human")
        sword = self._model("weapon-sword").copyTo(self.player.exposeJoint(None, "modelRoot", "arm-right"))
        sword.setPosHpr(-0.12, 0.0, -0.08, 0, 90, 0)
        self.pools = {k: [self._make_character(m) for _ in range(POOL_PER_KIND)] for k, m in ENEMY_MODELS.items()}
        for pool in self.pools.values():
            for a in pool:
                a.hide()
        self.potions = []

        self.tex = Texture()
        base.win.addRenderTexture(self.tex, GraphicsOutput.RTMCopyRam)

    # ---------- setup ----------
    def _model(self, name):
        if name not in self._models:
            self._models[name] = self.base.loader.loadModel(_asset(name))
        return self._models[name]

    def _make_character(self, name):
        a = Actor(_asset(name))
        a.reparentTo(self.base.render)
        return a

    def _setup_lights(self):
        r = self.base.render
        amb = AmbientLight("amb")
        amb.setColor((0.3, 0.3, 0.34, 1))
        sun = DirectionalLight("sun")
        sun.setColor((1.5, 1.42, 1.3, 1))
        sun_np = r.attachNewNode(sun)
        sun_np.setHpr(35, -55, 0)
        r.setLight(r.attachNewNode(amb))
        r.setLight(sun_np)
        simplepbr.init(render_node=r, window=self.base.win, camera_node=self.base.cam,
                       use_normal_maps=False, enable_shadows=False, use_emission_maps=False)

    def _setup_hud(self):
        cm = CardMaker("hp")
        cm.setFrame(0, 1, 0, 1)
        a2d = self.base.a2dTopLeft
        self.hp_bg = a2d.attachNewNode(cm.generate())
        self.hp_bg.setPos(0.05, 0, -0.14)
        self.hp_bg.setScale(0.62, 1, 0.09)
        self.hp_bg.setColor(0.15, 0.05, 0.05, 1)
        self.hp_fill = a2d.attachNewNode(cm.generate())
        self.hp_fill.setPos(0.06, 0, -0.13)
        self.hp_fill.setColor(0.85, 0.12, 0.12, 1)
        for n in (self.hp_bg, self.hp_fill):
            n.setTransparency(TransparencyAttrib.MNone)

    # ---------- per episode ----------
    def build(self, game: core.Game):
        if self.map_root is not None:
            self.map_root.removeNode()
        rng = random.Random(game.seed ^ 0x5EED)
        root = NodePath("map")
        for y in range(core.MAP_H):
            for x in range(core.MAP_W):
                cell = game.grid[y][x]
                floor = "floor-detail" if game.floor_var[y][x] else "floor"
                self._model(floor).copyTo(root).setPos(x + 0.5, y + 0.5, 0)
                if cell in PROP_MODEL:
                    m = self._model(PROP_MODEL[cell]).copyTo(root)
                    m.setPos(x + 0.5, y + 0.5, 0)
                    m.setH(rng.choice((0, 90, 180, 270)))
        root.flattenStrong()
        root.reparentTo(self.base.render)
        self.map_root = root
        for p in self.potions:
            p.removeNode()
        self.potions = []
        for _ in game.potions:
            p = self._model("potion").copyTo(self.base.render)
            p.setScale(1.6)
            self.potions.append(p)

    # ---------- per tick ----------
    def sync(self, game: core.Game):
        self._pose(self.player, game.player)
        slots = {k: iter(pool) for k, pool in self.pools.items()}
        for e in game.enemies:
            self._pose(next(slots[e.kind]), e)
        for it in slots.values():
            for a in it:
                a.hide()
        for node, pt in zip(self.potions, game.potions):
            if pt.active:
                node.show()
                node.setPos(pt.x, pt.y, 0)
            else:
                node.hide()
        p = game.player
        self.base.cam.setPos(p.x + CAM_OFFSET[0], p.y + CAM_OFFSET[1], CAM_OFFSET[2])
        self.base.cam.lookAt(p.x, p.y, 0.4)
        self.hp_fill.setScale(0.6 * max(p.hp, 0) / core.PLAYER_MAX_HP + 1e-4, 1, 0.07)

    def _pose(self, actor, ent):
        actor.show()
        actor.setPos(ent.x, ent.y, 0)
        actor.setH(math.degrees(ent.facing) + 90)
        anim = ANIM[ent.anim]
        n = actor.getNumFrames(anim)
        if ent.anim in ACTION_TICKS:  # stretch/squash clip to the action's duration
            ticks = ACTION_TICKS[ent.anim] if ent.kind == "player" else core.ENEMY_ATTACK_TICKS
            f = ent.anim_t * n // ticks
        else:
            f = ent.anim_t * core.ANIM_FPS // core.TICK_HZ
        actor.pose(anim, f % n if ent.anim in LOOPING else min(f, n - 1))
        if ent.flash:
            actor.setColorScale(2.0, 0.45, 0.45, 1)
        else:
            actor.clearColorScale()

    def grab(self):
        """Last rendered frame as HxWx3 uint8 (top row first)."""
        w, h = self.tex.getXSize(), self.tex.getYSize()
        buf = self.tex.getRamImageAs("RGB")
        if not buf:
            return np.zeros((h, w, 3), np.uint8)
        return np.frombuffer(memoryview(buf), np.uint8).reshape(h, w, 3)[::-1].copy()
