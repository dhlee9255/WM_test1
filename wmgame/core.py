"""Pure game logic. Fixed 15 Hz tick, no rendering dependencies."""
import math
import random
from collections import deque
from dataclasses import dataclass, field

TICK_HZ = 15
ANIM_FPS = 30
KEYS = ("w", "a", "s", "d", "attack", "pickup")

ZONE = 12  # 4x4 grid of zones: 9x9 lot + 3-wide streets, streets also run along the outer wall
ZONES = 4
MAP_W = MAP_H = ZONES * ZONE + 3
(FLOOR, WALL, ROCKS, STONES, COLUMN, BARREL, TABLE, CHAIR, CHEST, POT, WOOD, FENCE, IRON_FENCE, STONE_WALL,
 GRAVESTONE, CRYPT, PINE, LIGHTPOST, HAY, PUMPKIN, BENCH, COFFIN, PILLAR) = range(23)
SOLID = set(range(1, 23))
# walkable floor decorations (separate layer, no collision)
NO_DECO, ROAD, DEBRIS, DEBRIS_WOOD, GRAVE = range(5)

PLAYER_MAX_HP = 10
PLAYER_SPEED = 3.0 / TICK_HZ
RADIUS = 0.3


@dataclass(frozen=True)
class EnemyType:
    hp: int
    speed: float
    aggro: float
    cooldown: int
    wander_speed: float


ENEMY_TYPES = {
    "skeleton": EnemyType(hp=1, speed=2.3 / TICK_HZ, aggro=9.0, cooldown=11, wander_speed=0.8),
    "zombie": EnemyType(hp=2, speed=1.5 / TICK_HZ, aggro=7.0, cooldown=15, wander_speed=0.6),
}
SKELETON_RATIO = 0.6
ENEMY_ATTACK_RANGE = 0.9
ENEMY_ATTACK_TICKS, ENEMY_HIT_TICK = 7, 4
MAX_ENEMIES = 9
INITIAL_ENEMIES = 5
SPAWN_EVERY = 30

NUM_POTIONS = 6
POTION_HEAL = 3
POTION_RESPAWN = 60

ATTACK_TICKS, ATTACK_HIT_TICK, ATTACK_RANGE, ATTACK_ARC = 4, 2, 1.15, math.radians(70)
PICKUP_TICKS, PICKUP_HIT_TICK, PICKUP_RANGE = 6, 3, 0.9
DIE_TICKS = 20
CORPSE_TICKS = 15


@dataclass
class Actor:
    id: int
    x: float
    y: float
    hp: int
    kind: str = "player"
    facing: float = -math.pi / 2
    anim: str = "idle"
    anim_t: int = 0
    action_t: int = 0
    cooldown: int = 0
    flash: int = 0
    stun: int = 0
    dead_t: int = -1
    wander: tuple = (0.0, 0.0)
    wander_t: int = 0

    def set_anim(self, name):
        if self.anim != name:
            self.anim, self.anim_t = name, 0

    @property
    def alive(self):
        return self.hp > 0


@dataclass
class Potion:
    x: float
    y: float
    respawn: int = 0

    @property
    def active(self):
        return self.respawn == 0


# ---------- map prefabs (9x9 lots, local coords, y = north) ----------
class Lot:
    """Solid cells {(x,y): (kind, rot)} plus walkable decorations {(x,y): (deco, rot)}."""

    def __init__(self):
        self.cells, self.deco = {}, {}

    def put(self, x, y, kind, rot=0):
        self.cells[(x, y)] = (kind, rot)

    def lay(self, x, y, deco, rot=0):
        self.deco[(x, y)] = (deco, rot)

    def outline(self, kind):
        for x, y in _outline():
            self.put(x, y, kind, 90 if x in (0, 8) and y not in (0, 8) else 0)

    def gate(self, side, width=2):
        """Open the middle of one wall; returns the opened cells and the tiles just inside."""
        start = 4 - width // 2
        span = range(start, start + width)
        opened = {"s": [(i, 0) for i in span], "n": [(i, 8) for i in span],
                  "w": [(0, i) for i in span], "e": [(8, i) for i in span]}[side]
        step = {"s": (0, 1), "n": (0, -1), "w": (1, 0), "e": (-1, 0)}[side]
        inside = [(x + step[0], y + step[1]) for x, y in opened]
        self.clear(opened)
        return opened, inside

    def path(self, cells):
        for x, y in cells:
            self.lay(x, y, ROAD)

    def scatter(self, r, n, decos=None):
        decos = decos or (DEBRIS, DEBRIS, DEBRIS_WOOD)
        for _ in range(n):
            x, y = r.randrange(1, 8), r.randrange(1, 8)
            if (x, y) not in self.cells and (x, y) not in self.deco:
                self.lay(x, y, r.choice(decos), r.randrange(4) * 90)

    def clear(self, cells):
        for c in cells:
            self.cells.pop(c, None)


def _outline(n=9):
    return [(x, y) for x in range(n) for y in range(n) if x in (0, n - 1) or y in (0, n - 1)]


def lot_house(r):
    L = Lot()
    L.outline(WALL)
    L.put(4, 4, TABLE)
    L.put(3, 4, CHAIR, 270)
    L.put(5, 4, CHAIR, 90)
    for c in r.sample([(1, 1), (7, 1), (1, 7), (7, 7)], r.randint(2, 4)):
        L.put(*c, BARREL)
    L.put(*r.choice([(4, 7), (1, 4), (7, 4)]), CHEST)
    L.put(*r.choice([(2, 7), (6, 7), (1, 2), (7, 6)]), POT)
    if r.random() < 0.5:
        L.put(2, 2, TABLE)
        L.put(2, 3, CHAIR, 180)
    for side in r.sample("snwe", r.randint(1, 3)):
        opened, inside = L.gate(side)
        L.clear(inside)
        L.path(opened + inside)
    return L


def lot_storage(r):
    L = Lot()
    L.outline(WALL)
    for x in (1, 2, 3, 5, 6, 7):
        L.put(x, 7, r.choice([BARREL, BARREL, CHEST]))
    for y in (2, 3, 4, 5, 6):
        L.put(1, y, BARREL)
        L.put(7, y, r.choice([POT, CHEST]))
    L.put(4, 4, CHEST)
    L.put(4, 5, BARREL)
    opened, inside = L.gate("s")
    L.clear(inside)
    L.path(opened + inside)
    L.scatter(r, 2, (DEBRIS_WOOD,))
    return L


def lot_hall(r):
    """Chapel: two colonnades, benches facing an altar."""
    L = Lot()
    for x in (1, 7):
        for y in (1, 3, 5, 7):
            L.put(x, y, COLUMN)
    for y in (2, 4):
        L.put(3, y, BENCH)
        L.put(5, y, BENCH)
    L.put(4, 7, TABLE)
    L.put(3, 7, PILLAR)
    L.put(5, 7, PILLAR)
    L.path([(4, y) for y in range(0, 7)])
    L.scatter(r, 3)
    return L


def lot_ruins(r):
    L = Lot()
    L.outline(WALL)
    walls = list(L.cells)
    for _ in range(r.randint(3, 5)):  # knock out chunks of wall, leave rubble
        x, y = r.choice(walls)
        L.clear([(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
        L.lay(x, y, DEBRIS, r.randrange(4) * 90)
        if r.random() < 0.5:
            L.put(x, y, STONES)
    L.put(r.randint(2, 6), r.randint(2, 6), ROCKS)
    L.put(r.randint(2, 6), r.randint(2, 6), PILLAR)
    L.scatter(r, 6)
    return L


def lot_garden(r):
    """Rock garden: boulder clusters, a couple of trees, pumpkins."""
    L = Lot()
    for cx, cy in r.sample([(1, 1), (6, 1), (1, 6), (6, 6)], 3):
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            if r.random() < 0.8:
                L.put(cx + dx, cy + dy, r.choice([ROCKS, ROCKS, STONES]))
    for x, y in r.sample([(4, 1), (1, 4), (7, 4), (4, 7)], 2):
        L.put(x, y, PINE)
    L.put(4, 4, PUMPKIN)
    L.scatter(r, 5)
    return L


def lot_market(r):
    """Crossroads market: paved cross, lamp posts at corners, four stalls."""
    L = Lot()
    for c in [(0, 0), (8, 0), (0, 8), (8, 8)]:
        L.put(*c, LIGHTPOST)
    L.path([(4, i) for i in range(9)] + [(i, 4) for i in range(9)])
    for cx, cy in [(2, 2), (6, 2), (2, 6), (6, 6)]:
        L.put(cx, cy, TABLE)
        L.put(cx + (-1 if cx < 4 else 1), cy, r.choice([BARREL, POT, CHEST, HAY]))
        if r.random() < 0.5:
            L.put(cx, cy + (1 if cy > 4 else -1), r.choice([BARREL, PUMPKIN]))
    return L


def lot_yard(r):
    """Fenced wood yard: timber racks, hay, pumpkins."""
    L = Lot()
    L.outline(FENCE)
    for y in (2, 4, 6):
        L.put(1, y, WOOD)
        L.put(2, y, WOOD)
    L.put(6, 6, HAY)
    L.put(7, 6, HAY)
    L.put(7, 5, HAY)
    L.put(6, 2, PUMPKIN)
    L.put(7, 2, PUMPKIN)
    opened, inside = L.gate("s")
    L.clear(inside)
    L.path(opened + inside + [(4, 2), (4, 3)])
    L.scatter(r, 5, (DEBRIS_WOOD,))
    return L


def lot_graveyard(r):
    """Iron-fenced graveyard: central path, rows of gravestones with graves in front."""
    L = Lot()
    L.outline(IRON_FENCE)
    opened, _ = L.gate("s", 3)
    for x in (1, 2, 6, 7):
        for y in (3, 5, 7):
            if r.random() < 0.9:
                L.put(x, y, GRAVESTONE)
                L.lay(x, y - 1, GRAVE)
    L.put(2, 1, LIGHTPOST)
    L.put(6, 1, LIGHTPOST)
    L.put(4, 7, CRYPT)
    L.path(opened + [(4, y) for y in range(1, 7)])
    return L


def lot_crypt(r):
    """Walled crypt yard: crypt between pillars, coffins, rubble."""
    L = Lot()
    L.outline(STONE_WALL)
    walls = [c for c in L.cells if c[1] != 0]
    L.clear(r.sample(walls, r.randint(1, 2)))  # broken wall sections
    opened, _ = L.gate("s", 3)
    L.put(4, 5, CRYPT)
    L.put(3, 5, PILLAR)
    L.put(5, 5, PILLAR)
    L.put(2, 2, COFFIN, 90)
    L.put(6, 2, COFFIN, 90)
    L.put(1, 7, GRAVESTONE)
    L.put(7, 7, GRAVESTONE)
    L.path(opened + [(4, 1), (4, 2), (4, 3), (4, 4)])
    L.scatter(r, 5)
    return L


def lot_forest(r):
    L = Lot()
    for x, y in r.sample([(1, 1), (1, 4), (1, 7), (4, 1), (4, 7), (7, 1), (7, 4), (7, 7)], 5):
        L.put(x, y, PINE)
    for _ in range(2):
        x, y = r.randint(2, 6), r.randint(2, 6)
        if (x, y) != (4, 4):
            L.put(x, y, r.choice([ROCKS, STONES, PUMPKIN]))
    L.scatter(r, 6)
    return L


LOTS = [(lot_house, 3), (lot_storage, 2), (lot_hall, 2), (lot_ruins, 2), (lot_garden, 1), (lot_market, 2),
        (lot_yard, 2), (lot_graveyard, 3), (lot_crypt, 2), (lot_forest, 1)]


def _rotate(lot, k, n=9):
    """Rotate a lot k*90 degrees counter-clockwise (cells and their headings)."""
    cells, deco = lot.cells, lot.deco
    for _ in range(k):
        cells = {(n - 1 - y, x): (v, (rot + 90) % 360) for (x, y), (v, rot) in cells.items()}
        deco = {(n - 1 - y, x): (v, (rot + 90) % 360) for (x, y), (v, rot) in deco.items()}
    return cells, deco


@dataclass
class Game:
    seed: int
    rng: random.Random = field(init=False)
    grid: list = field(init=False)
    rot: list = field(init=False)
    deco: list = field(init=False)
    deco_rot: list = field(init=False)
    floor_var: list = field(init=False)
    player: Actor = field(init=False)
    enemies: list = field(init=False)
    potions: list = field(init=False)
    tick: int = 0
    kills: int = 0
    done: bool = False
    _next_id: int = 1

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self._gen_map()
        cx, cy = MAP_W // 2 + 0.5, MAP_H // 2 + 0.5
        self.player = Actor(0, cx, cy, PLAYER_MAX_HP)
        self.enemies = []
        self.potions = [Potion(*self._random_floor(min_dist=4)) for _ in range(NUM_POTIONS)]
        for _ in range(INITIAL_ENEMIES):
            self._spawn_enemy()

    # ---------- map ----------
    def _gen_map(self):
        r = self.rng
        g = [[FLOOR] * MAP_W for _ in range(MAP_H)]
        rot = [[0] * MAP_W for _ in range(MAP_H)]
        deco = [[NO_DECO] * MAP_W for _ in range(MAP_H)]
        deco_rot = [[0] * MAP_W for _ in range(MAP_H)]
        for i in range(MAP_W):
            g[0][i] = g[MAP_H - 1][i] = WALL
        for j in range(MAP_H):
            g[j][0] = g[j][MAP_W - 1] = WALL
        # paved roads down the middle of every street between lots
        for i in range(1, ZONES):
            c = i * ZONE + 1
            for k in range(1, MAP_W - 1):
                deco[k][c] = deco[c][k] = ROAD
        fns, weights = zip(*LOTS)
        for zy in range(ZONES):
            for zx in range(ZONES):
                lot = r.choices(fns, weights)[0](r)
                cells, lot_deco = _rotate(lot, r.randrange(4))
                ox, oy = zx * ZONE + 3, zy * ZONE + 3
                for (x, y), (kind, a) in cells.items():
                    g[oy + y][ox + x], rot[oy + y][ox + x] = kind, a
                for (x, y), (kind, a) in lot_deco.items():
                    deco[oy + y][ox + x], deco_rot[oy + y][ox + x] = kind, a
        for i in range(1, ZONES):  # lamp posts on crossroad corners, barrels lining some streets
            for j in range(1, ZONES):
                if r.random() < 0.5:
                    x, y = i * ZONE + r.choice((0, 2)), j * ZONE + r.choice((0, 2))
                    g[y][x] = LIGHTPOST
            for j in range(ZONES):
                if r.random() < 0.25:
                    x, y = i * ZONE + 2, j * ZONE + r.choice((4, 9))
                    g[y][x] = g[y + 1][x] = BARREL
        # street furniture on the side lanes, only against a solid lot edge (never in front of a gate)
        props = (BARREL, BARREL, HAY, PUMPKIN, BENCH, GRAVESTONE, PINE, LIGHTPOST)
        for i in range(1, ZONES):
            for side, lot_dir in ((i * ZONE, -1), (i * ZONE + 2, 1)):
                for k in range(3, MAP_H - 3):
                    if k % ZONE in (0, 1, 2) or r.random() > 0.3:
                        continue
                    kind = r.choice(props)
                    if g[k][side] == FLOOR and g[k][side + lot_dir] in SOLID:  # vertical street
                        g[k][side], rot[k][side] = kind, 90
                    kind = r.choice(props)
                    if g[side][k] == FLOOR and g[side + lot_dir][k] in SOLID:  # horizontal street
                        g[side][k], rot[side][k] = kind, 0
        for y in range(1, MAP_H - 1):  # loose rubble on open ground
            for x in range(1, MAP_W - 1):
                if g[y][x] == FLOOR and deco[y][x] == NO_DECO and r.random() < 0.06:
                    deco[y][x], deco_rot[y][x] = r.choice((DEBRIS, DEBRIS, DEBRIS_WOOD)), r.randrange(4) * 90
        # every floor tile must be reachable from spawn
        c = MAP_W // 2
        g[c][c] = FLOOR
        seen = [[False] * MAP_W for _ in range(MAP_H)]
        q = deque([(c, c)])
        seen[c][c] = True
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not seen[ny][nx] and g[ny][nx] not in SOLID:
                    seen[ny][nx] = True
                    q.append((nx, ny))
        self.unreachable = 0
        for y in range(MAP_H):
            for x in range(MAP_W):
                if g[y][x] == FLOOR and not seen[y][x]:
                    g[y][x] = STONES
                    self.unreachable += 1
        self.grid, self.rot, self.deco, self.deco_rot = g, rot, deco, deco_rot
        self.floor_var = [[1 if r.random() < 0.15 else 0 for _ in range(MAP_W)] for _ in range(MAP_H)]

    def solid(self, tx, ty):
        if tx < 0 or ty < 0 or tx >= MAP_W or ty >= MAP_H:
            return True
        return self.grid[ty][tx] in SOLID

    def _collides(self, x, y):
        for ty in range(int(y - RADIUS), int(y + RADIUS) + 1):
            for tx in range(int(x - RADIUS), int(x + RADIUS) + 1):
                if self.solid(tx, ty):
                    return True
        return False

    def _move(self, a, dx, dy):
        if dx and not self._collides(a.x + dx, a.y):
            a.x += dx
        if dy and not self._collides(a.x, a.y + dy):
            a.y += dy

    def _random_floor(self, min_dist=0.0, max_dist=1e9):
        p = getattr(self, "player", None)
        for _ in range(500):
            x, y = self.rng.randrange(1, MAP_W - 1), self.rng.randrange(1, MAP_H - 1)
            if self.grid[y][x] != FLOOR:
                continue
            if p is not None:
                d = math.hypot(x + 0.5 - p.x, y + 0.5 - p.y)
                if d < min_dist or d > max_dist:
                    continue
            return x + 0.5, y + 0.5
        return MAP_W // 2 + 0.5, MAP_H // 2 + 0.5

    def _spawn_enemy(self):
        x, y = self._random_floor(min_dist=7, max_dist=13)
        kind = "skeleton" if self.rng.random() < SKELETON_RATIO else "zombie"
        self.enemies.append(Actor(self._next_id, x, y, ENEMY_TYPES[kind].hp, kind=kind))
        self._next_id += 1

    # ---------- step ----------
    def step(self, keys):
        """keys: sequence of 6 bools in KEYS order. Returns info dict."""
        info = {"kills": 0, "damage": 0, "heal": 0}
        if self.done:
            return info
        self.tick += 1
        self._step_player(keys, info)
        for e in self.enemies:
            self._step_enemy(e, info)
        self._separate_enemies()
        self.enemies = [e for e in self.enemies if e.alive or e.dead_t < CORPSE_TICKS]
        if self.tick % SPAWN_EVERY == 0 and sum(e.alive for e in self.enemies) < MAX_ENEMIES:
            self._spawn_enemy()
        for pt in self.potions:
            if pt.respawn > 0:
                pt.respawn -= 1
                if pt.respawn == 0:
                    pt.x, pt.y = self._random_floor(min_dist=5)
        self.kills += info["kills"]
        return info

    def _step_player(self, keys, info):
        p = self.player
        p.anim_t += 1
        p.flash = max(0, p.flash - 1)
        if not p.alive:
            p.dead_t += 1
            if p.dead_t >= DIE_TICKS:
                self.done = True
            return
        w, a, s, d, atk, pick = keys
        if p.anim in ("attack", "pickup"):
            p.action_t += 1
            if p.anim == "attack" and p.action_t == ATTACK_HIT_TICK:
                self._player_hit(info)
            if p.anim == "pickup" and p.action_t == PICKUP_HIT_TICK:
                self._player_pickup(info)
            if p.action_t >= (ATTACK_TICKS if p.anim == "attack" else PICKUP_TICKS):
                p.set_anim("idle")
            return
        if atk:
            p.set_anim("attack")
            p.action_t = 0
            return
        if pick:
            p.set_anim("pickup")
            p.action_t = 0
            return
        dx, dy = (d - a), (w - s)
        if dx or dy:
            n = math.hypot(dx, dy)
            p.facing = math.atan2(dy, dx)
            self._move(p, dx / n * PLAYER_SPEED, dy / n * PLAYER_SPEED)
            p.set_anim("walk")
        else:
            p.set_anim("idle")

    def _player_hit(self, info):
        p = self.player
        for e in self.enemies:
            if not e.alive:
                continue
            dx, dy = e.x - p.x, e.y - p.y
            dist = math.hypot(dx, dy)
            ang = abs((math.atan2(dy, dx) - p.facing + math.pi) % (2 * math.pi) - math.pi)
            if dist <= ATTACK_RANGE and (ang <= ATTACK_ARC or dist < 0.4):
                e.hp -= 1
                e.flash, e.stun = 3, 4
                if dist > 1e-6:
                    self._move(e, dx / dist * 0.35, dy / dist * 0.35)
                if not e.alive:
                    e.set_anim("die")
                    e.dead_t = 0
                    info["kills"] += 1

    def _player_pickup(self, info):
        p = self.player
        for pt in self.potions:
            if pt.active and math.hypot(pt.x - p.x, pt.y - p.y) <= PICKUP_RANGE:
                heal = min(POTION_HEAL, PLAYER_MAX_HP - p.hp)
                p.hp += heal
                info["heal"] += heal
                pt.respawn = POTION_RESPAWN
                break

    def _step_enemy(self, e, info):
        t = ENEMY_TYPES[e.kind]
        e.anim_t += 1
        e.flash = max(0, e.flash - 1)
        e.cooldown = max(0, e.cooldown - 1)
        if not e.alive:
            e.dead_t += 1
            return
        p = self.player
        if e.stun > 0:
            e.stun -= 1
            if e.anim == "attack":
                e.set_anim("idle")
            return
        dx, dy = p.x - e.x, p.y - e.y
        dist = math.hypot(dx, dy)
        if e.anim == "attack":
            e.action_t += 1
            if e.action_t == ENEMY_HIT_TICK and p.alive and dist <= ENEMY_ATTACK_RANGE + 0.25:
                p.hp -= 1
                p.flash = 3
                info["damage"] += 1
                if not p.alive:
                    p.set_anim("die")
                    p.dead_t = 0
            if e.action_t >= ENEMY_ATTACK_TICKS:
                e.set_anim("idle")
                e.cooldown = t.cooldown
            return
        if p.alive and dist <= ENEMY_ATTACK_RANGE:
            e.facing = math.atan2(dy, dx)
            if e.cooldown == 0:
                e.set_anim("attack")
                e.action_t = 0
            else:
                e.set_anim("idle")
            return
        if p.alive and dist <= t.aggro:
            e.facing = math.atan2(dy, dx)
            self._move(e, dx / dist * t.speed, dy / dist * t.speed)
            e.set_anim("walk")
            return
        if e.wander_t <= 0:
            if self.rng.random() < 0.5:
                e.wander = (0.0, 0.0)
            else:
                ang = self.rng.uniform(-math.pi, math.pi)
                e.wander = (math.cos(ang), math.sin(ang))
            e.wander_t = self.rng.randint(10, 30)
        e.wander_t -= 1
        wx, wy = e.wander
        if wx or wy:
            e.facing = math.atan2(wy, wx)
            self._move(e, wx * t.speed * t.wander_speed, wy * t.speed * t.wander_speed)
            e.set_anim("walk")
        else:
            e.set_anim("idle")

    def _separate_enemies(self):
        alive = [e for e in self.enemies if e.alive]
        for i, a in enumerate(alive):
            for b in alive[i + 1:]:
                dx, dy = b.x - a.x, b.y - a.y
                d = math.hypot(dx, dy)
                if 1e-6 < d < 0.6:
                    push = (0.6 - d) / 2
                    self._move(a, -dx / d * push, -dy / d * push)
                    self._move(b, dx / d * push, dy / d * push)
