"""Batched world fields. One private room per slot.

No model of the room lives here. The only model of the room is inside an agent.
"""

import numpy as np
import torch


class Fields:
    def __init__(self, cfg, n: int):
        self.cfg = cfg
        self.N = n
        self.G = cfg.grid
        self.ar = torch.arange(n)
        self.angles = torch.arange(8, dtype=torch.float32) * (math_pi() / 4.0)
        self.ds = torch.linspace(0.4, cfg.max_range, cfg.ray_steps)

        self.walls = torch.zeros(n, self.G, self.G, dtype=torch.bool)
        self.fertility = torch.zeros(n, self.G, self.G, dtype=torch.float32)
        self.food = torch.zeros(n, self.G, self.G, dtype=torch.float32)

    # ------------------------------------------------------------------ make
    def new(self, slot: int, rng: np.random.Generator, density: float | None = None) -> None:
        G = self.G
        walls = np.zeros((G, G), dtype=bool)
        walls[0, :] = walls[-1, :] = True
        walls[:, 0] = walls[:, -1] = True
        for _ in range(G // 5):
            x0 = int(rng.integers(2, G - 4))
            y0 = int(rng.integers(2, G - 4))
            w = int(rng.integers(1, 4))
            h = int(rng.integers(1, 4))
            walls[y0:y0 + h, x0:x0 + w] = True

        d = self.cfg.fertility if density is None else max(0.0, min(0.5, float(density)))
        fert = rng.random((G, G)) < d
        fert[walls] = False
        food = fert * rng.uniform(0.4, 1.0, (G, G))

        self.walls[slot] = torch.from_numpy(walls)
        self.fertility[slot] = torch.from_numpy(fert.astype(np.float32))
        self.food[slot] = torch.from_numpy(food.astype(np.float32))

    def random_pos(self, slot: int, rng: np.random.Generator):
        G = self.G
        while True:
            x = float(rng.uniform(1.5, G - 2.5))
            y = float(rng.uniform(1.5, G - 2.5))
            if not self.walls[slot, int(y), int(x)]:
                break
        heading = float(rng.uniform(-np.pi, np.pi))
        return x, y, heading

    # ----------------------------------------------------------------- sense
    def _food_at(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        G = self.G
        ix = xs.long().clamp(0, G - 1)
        iy = ys.long().clamp(0, G - 1)
        return self.food[self.ar, iy, ix]

    def sense(self, x, y, heading, energy) -> torch.Tensor:
        G = self.G
        N = self.N

        ang = heading[:, None] + self.angles[None, :]                  # [N,8]
        ds = self.ds
        xs = x[:, None, None] + torch.cos(ang)[:, :, None] * ds[None, None, :]
        ys = y[:, None, None] + torch.sin(ang)[:, :, None] * ds[None, None, :]
        ix = xs.long().clamp(0, G - 1)
        iy = ys.long().clamp(0, G - 1)

        hit = self.walls[self.ar[:, None, None], iy, ix]               # [N,8,K]
        any_hit = hit.any(dim=2)
        first = hit.float().argmax(dim=2)
        dist = torch.where(any_hit, ds[first], torch.full_like(ds[first], self.cfg.max_range))
        whisk = 1.0 - dist / self.cfg.max_range                        # [N,8]

        cx = x.long().clamp(0, G - 1)
        cy = y.long().clamp(0, G - 1)
        food_here = self.food[self.ar, cy, cx]

        d = 0.7
        gx = (self._food_at(x + d, y) - self._food_at(x - d, y)) * 3.0
        gy = (self._food_at(x, y + d) - self._food_at(x, y - d)) * 3.0
        gx = gx.clamp(-1.0, 1.0)
        gy = gy.clamp(-1.0, 1.0)

        sin = torch.sin(heading)
        cos = torch.cos(heading)
        # same food direction, but in the body frame: ahead and to the left
        fwd = (gx * cos + gy * sin).clamp(-1.0, 1.0)
        left = (-gx * sin + gy * cos).clamp(-1.0, 1.0)
        en = (energy / 100.0).clamp(0.0, 2.0)

        return torch.cat([
            whisk,
            food_here[:, None],
            fwd[:, None],
            left[:, None],
            sin[:, None],
            cos[:, None],
            en[:, None],
        ], dim=1)

    # ------------------------------------------------------------------ step
    def apply(self, actions, x, y, heading, energy, eaten):
        cfg = self.cfg
        G = self.G

        turn = torch.where(
            actions == 1, torch.full_like(heading, -cfg.turn_rate),
            torch.where(actions == 2, torch.full_like(heading, cfg.turn_rate),
                        torch.zeros_like(heading)),
        )
        heading = torch.atan2(torch.sin(heading + turn), torch.cos(heading + turn))

        move = actions == 0
        nx = x + torch.cos(heading) * cfg.move_speed * move
        ny = y + torch.sin(heading) * cfg.move_speed * move
        ci = nx.long().clamp(0, G - 1)
        cj = ny.long().clamp(0, G - 1)
        blocked = self.walls[self.ar, cj, ci]
        go = move & (~blocked)
        x = torch.where(go, nx, x)
        y = torch.where(go, ny, y)

        ex = x.long().clamp(0, G - 1)
        ey = y.long().clamp(0, G - 1)
        here = self.food[self.ar, ey, ex]
        take = torch.where(actions == 3, torch.minimum(here, torch.ones_like(here)),
                           torch.zeros_like(here))
        self.food[self.ar, ey, ex] -= take
        energy = energy + take * cfg.bite
        eaten = eaten + (take > 0.01).float()

        return x, y, heading, energy, eaten

    def regen(self) -> None:
        # food returns toward fertility. slow.
        torch.minimum(self.food + self.cfg.food_regen * self.fertility,
                      self.fertility, out=self.food)


def math_pi() -> float:
    return 3.141592653589793
