"""Scripted bot with full map knowledge. Hunts enemies along planned paths, roams the map when
none are near, drinks potions, and occasionally swings at nothing."""
import math
import random
from collections import deque

from . import core

NOOP = (0, 0, 0, 0, 0, 0)
ATTACK = (0, 0, 0, 0, 1, 0)
PICKUP = (0, 0, 0, 0, 0, 1)
DEADZONE = 0.12
HUNT_RADIUS = 20


def steer(dx, dy):
    """Keys that move toward (dx, dy); axes within the deadzone are left alone."""
    w, s = dy > DEADZONE, dy < -DEADZONE
    d, a = dx > DEADZONE, dx < -DEADZONE
    return (int(w), int(a), int(s), int(d), 0, 0)


DIR8 = [(0, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 0), (1, 1, 0, 0),  # E, NE, N, NW (w,a,s,d)
        (0, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 0), (0, 0, 1, 1)]  # W, SW, S, SE


class PassiveBot:
    """Stands still and never attacks, so enemies walk up and kill it (death-sequence data)."""

    def act(self, g):
        return NOOP


class ScriptedBot:
    """smooth=True: no random/hesitant actions, line-of-sight path shortcuts, stable 8-way keys."""

    def __init__(self, seed, eps=0.04, smooth=False):
        self.rng = random.Random(seed)
        self.smooth = smooth
        self.eps = 0.0 if smooth else eps
        self.prev_dir = None
        self.aggression = self.rng.uniform(0.7, 1.0)
        self.heal_at = self.rng.choice([0, 0, 3, 4, 5])  # 0 = never drinks, so some episodes end in death
        self.hold, self.held = 0, NOOP
        self.goal = None       # roaming destination (tile)
        self.path = []         # tiles from current position to self.path_goal
        self.path_goal = None
        self.last_pos, self.stuck = None, 0

    # ---------- main policy ----------
    def act(self, g: core.Game):
        r = self.rng
        if self.hold > 0:
            self.hold -= 1
            return self.held
        p = g.player
        self._update_stuck(p)
        if self.stuck > 10:  # wedged on something: jiggle, then go somewhere else
            self.stuck, self.goal, self.path = 0, None, []
            return self._commit(steer(r.uniform(-1, 1), r.uniform(-1, 1)), r.randint(2, 5))
        if r.random() < self.eps:
            return self._commit(self._random_action(), r.randint(0, 3))

        alive = [e for e in g.enemies if e.alive]
        enemy = min(alive, key=lambda e: math.hypot(e.x - p.x, e.y - p.y), default=None)
        enemy_d = math.hypot(enemy.x - p.x, enemy.y - p.y) if enemy else 1e9
        pots = [pt for pt in g.potions if pt.active]
        pot = min(pots, key=lambda pt: math.hypot(pt.x - p.x, pt.y - p.y), default=None)
        pot_d = math.hypot(pot.x - p.x, pot.y - p.y) if pot else 1e9

        wants_heal = self.heal_at > 0 and (p.hp <= self.heal_at or (p.hp < core.PLAYER_MAX_HP and pot_d < 3))
        if wants_heal and pot and pot_d < 14 and not (enemy_d < 1.2 and r.random() < self.aggression):
            if pot_d < 0.6:
                return self._commit(PICKUP, 0)
            return self._go(g, pot.x, pot.y)

        if enemy and enemy_d < 5:
            dx, dy = enemy.x - p.x, enemy.y - p.y
            if enemy_d < 1.0:
                ang = abs((math.atan2(dy, dx) - p.facing + math.pi) % (2 * math.pi) - math.pi)
                if ang < math.radians(45):
                    return self._commit(ATTACK, r.randint(0, 1))
                return steer(dx, dy)  # turn to face it
            if not self.smooth and enemy_d < 1.8 and r.random() < 0.06:
                return self._commit(ATTACK, 0)  # swing too early and miss
            if not self.smooth and enemy_d < 2.5 and r.random() > self.aggression:
                return self._commit(steer(-dx, -dy), r.randint(2, 5))  # back off
            return self._go(g, enemy.x, enemy.y)

        air_swing = 0.006 if self.smooth else 0.012
        if r.random() < air_swing:
            return self._commit(ATTACK, 0 if self.smooth else r.choice((0, 0, 0, 4)))  # swing at nothing
        if enemy and enemy_d < HUNT_RADIUS:
            return self._go(g, enemy.x, enemy.y)  # knows the map: hunt it down
        if not self.smooth and r.random() < 0.005:
            return self._commit(NOOP, r.randint(3, 12))  # brief pause
        return self._roam(g)

    # ---------- helpers ----------
    def _commit(self, keys, extra):
        self.held, self.hold = tuple(keys), extra
        return self.held

    def _random_action(self):
        roll = self.rng.random()
        if roll < 0.12:
            return ATTACK
        if roll < 0.2:
            return PICKUP
        if roll < 0.3:
            return NOOP
        return steer(self.rng.uniform(-1, 1), self.rng.uniform(-1, 1))

    def _update_stuck(self, p):
        pos = (p.x, p.y)
        moving = self.held[:4] != (0, 0, 0, 0)
        if self.last_pos and moving and math.hypot(pos[0] - self.last_pos[0], pos[1] - self.last_pos[1]) < 0.02:
            self.stuck += 1
        else:
            self.stuck = max(0, self.stuck - 1)
        self.last_pos = pos

    def _roam(self, g):
        p = g.player
        here = (int(p.x), int(p.y))
        if self.goal is None or here == self.goal:
            self.goal = self._far_tile(g, here)
        return self._go(g, self.goal[0] + 0.5, self.goal[1] + 0.5)

    def _far_tile(self, g, here):
        """Random reachable floor tile, biased toward far away so the bot crosses the map."""
        best, best_d = here, -1.0
        for _ in range(6):
            x, y = self.rng.randrange(1, core.MAP_W - 1), self.rng.randrange(1, core.MAP_H - 1)
            if g.grid[y][x] != core.FLOOR:
                continue
            d = math.hypot(x - here[0], y - here[1])
            if d > best_d:
                best, best_d = (x, y), d
        return best

    def _go(self, g, tx, ty):
        """Follow a cached BFS path toward (tx, ty); replan when the goal tile changes or we leave the path."""
        p = g.player
        here, goal = (int(p.x), int(p.y)), (int(tx), int(ty))
        if here == goal:
            self.held = steer(tx - p.x, ty - p.y)
            return self.held
        if goal != self.path_goal or here not in self.path:
            self.path, self.path_goal = self._bfs(g, here, goal), goal
            if not self.path:
                self.goal = None
                self.held = steer(tx - p.x, ty - p.y)
                return self.held
        i = self.path.index(here)
        self.path = self.path[i:]
        nxt = self.path[1]
        if self.smooth:
            for node in self.path[2:12]:  # shortcut to the farthest node in straight line of sight
                if not self._clear(g, p.x, p.y, node[0] + 0.5, node[1] + 0.5):
                    break
                nxt = node
            self.held = self._dir_keys(nxt[0] + 0.5 - p.x, nxt[1] + 0.5 - p.y)
        else:
            self.held = steer(nxt[0] + 0.5 - p.x, nxt[1] + 0.5 - p.y)
        return self.held

    @staticmethod
    def _clear(g, x0, y0, x1, y1):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.2))
        return not any(g._collides(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(1, n + 1))

    def _dir_keys(self, dx, dy):
        """8-way keys with hysteresis: keep the current direction unless the heading moved > 30 deg."""
        ang = math.atan2(dy, dx)
        k = round(ang / (math.pi / 4)) % 8
        if self.prev_dir is not None:
            diff = abs((ang - self.prev_dir * math.pi / 4 + math.pi) % (2 * math.pi) - math.pi)
            if diff < math.radians(30):
                k = self.prev_dir
        self.prev_dir = k
        return DIR8[k] + (0, 0)

    def _bfs(self, g, start, goal):
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.rng.shuffle(dirs)  # random tie-breaking once per plan (not per tick) avoids direction bias
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for dx, dy in dirs:
                n = (cur[0] + dx, cur[1] + dy)
                if n not in prev and not g.solid(*n):
                    prev[n] = cur
                    q.append(n)
        if goal not in prev:
            return []
        path, node = [], goal
        while node is not None:
            path.append(node)
            node = prev[node]
        return path[::-1]
