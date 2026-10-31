"""The population. No reward anywhere.

Energy falls. Food raises it. At zero the body stops. Above a threshold the
body splits. The child inherits the weights, with noise. The urge to eat is
not stored. It is what remains of the lines that did not stop.
"""

from collections import deque
from pathlib import Path

import math
import numpy as np
import torch

from .brain import BatchedBrains
from .names import NAMES
from .world import Fields

MOVE, LEFT, RIGHT, EAT = 0, 1, 2, 3


class Population:
    def __init__(self, cfg, seed: int, logger, init: bool = True):
        self.cfg = cfg
        self.log = logger
        self.rng = np.random.default_rng(seed)
        self.N = cfg.n_slots
        self.ar = torch.arange(self.N)
        self.competitive_birth = True
        self.birth_mode = "threshold"      # weakest | threshold | oldest | off
        self.birth_threshold = 15.0
        self.min_replace_age = 200
        self.max_age = 0                  # 0 = off
        self.food_density = cfg.fertility
        self.base_cap = 100.0
        self.cap_growth = 0.00001
        self.energy_cap = torch.full((self.N,), self.base_cap)
        self.action_score = torch.zeros(self.N, cfg.act_dim)
        self.min_pop = cfg.min_pop
        self.cycle_on = False
        self.cycle_low = 0.006
        self.cycle_high = 0.03
        self.cycle_period = 6000
        self.cycle_phase = 0.0
        self.cycle_step = 100
        self.sotf_enabled = False
        self.sotf_interval = 0
        self.sotf_last = 0
        self.sotf_session = 0
        self.sotf_tier = torch.full((self.N,), -1, dtype=torch.long)
        self.visited = torch.zeros(self.N, cfg.grid, cfg.grid, dtype=torch.bool)
        self.path = torch.zeros(self.N)
        self.obs_prev = torch.zeros(self.N, cfg.obs_dim)
        self.seat_stats = [self._fresh_seat_stats() for _ in range(self.N)]
        self.max_gen_saved = 0

        self.brains = BatchedBrains(cfg)
        self.fields = Fields(cfg, self.N)
        # Use-dependent inheritance. The policy is frozen inside a life. The
        # parent records the decayed average of effective use. The child gets
        # the record at birth. The gain matches the old 0.001-per-step total
        # over the ~2,000-step decay horizon.
        self.use_gain = 2.0
        self.use_decay = 0.9995
        self.use = torch.zeros(self.N, cfg.hidden, cfg.act_dim)

        self.alive = torch.zeros(self.N, dtype=torch.bool)
        self.ids = torch.full((self.N,), -1, dtype=torch.long)
        self.parent = torch.full((self.N,), -1, dtype=torch.long)
        self.gen = torch.zeros(self.N, dtype=torch.long)
        self.age = torch.zeros(self.N, dtype=torch.long)
        self.energy = torch.zeros(self.N)
        self.x = torch.zeros(self.N)
        self.y = torch.zeros(self.N)
        self.heading = torch.zeros(self.N)
        self.h = torch.zeros(self.N, cfg.hidden)
        self.prev = torch.zeros(self.N, dtype=torch.long)
        self.eaten = torch.zeros(self.N)
        self.pred_err = torch.zeros(self.N)
        self.obs = torch.zeros(self.N, cfg.obs_dim)
        self.probs = torch.full((self.N, cfg.act_dim), 0.25)
        self.actions = torch.zeros(self.N, dtype=torch.long)
        self.salience = torch.zeros(self.N, 6)

        self.next_id = 0
        self.step_i = 0
        self.births = 0
        self.deaths = 0
        self.reseeds = 0
        self.best_eaten = 0.0
        self.best_life = 0
        self.epoch_step = 0
        self.lifetimes = deque(maxlen=300)
        self.accum = 0
        self.accum_steps = 4

        for s in range(cfg.init_pop):
            if init:
                self.spawn(s, None)

        # founders are not births in the population count
        self.births = 0

    @staticmethod
    def _fresh_seat_stats() -> dict:
        return {
            "births": 0, "deaths": 0, "total_life": 0, "max_life": 0,
            "total_eaten": 0.0, "best_eaten": 0.0,
            "gen_sum": 0, "gen_count": 0, "last_id": -1,
        }

    # ----------------------------------------------------------------- birth
    def spawn(self, slot: int, parent_slot: int | None) -> None:
        cfg = self.cfg
        if parent_slot is None:
            self.brains.randomize(slot)
            gen = 0
            pid = -1
        else:
            self.brains.copy(parent_slot, slot)
            with torch.no_grad():
                self.brains.W_pol[slot].add_(self.use_gain * self.use[parent_slot])
                self.brains.W_pol[slot].clamp_(-2.0, 2.0)
            self.brains.mutate(slot)
            gen = int(self.gen[parent_slot]) + 1
            pid = int(self.ids[parent_slot])

        self.alive[slot] = True
        self.ids[slot] = self.next_id
        self.next_id += 1
        self.parent[slot] = pid
        self.gen[slot] = gen
        self.age[slot] = 0
        self.energy[slot] = cfg.start_energy if parent_slot is None else cfg.child_energy
        self.energy_cap[slot] = self.base_cap
        if parent_slot is None:
            self.action_score[slot] = torch.randn(cfg.act_dim) * 0.01
        else:
            # the child gets half of the parent's action score, plus noise
            self.action_score[slot] = (0.5 * self.action_score[parent_slot]
                                       + torch.randn(cfg.act_dim) * 0.02)
        self.fields.new(slot, self.rng, self.food_density)
        self.x[slot], self.y[slot], self.heading[slot] = self.fields.random_pos(slot, self.rng)
        self.h[slot].zero_()
        self.prev[slot] = 0
        self.eaten[slot] = 0.0
        self.pred_err[slot] = 0.0
        self.actions[slot] = 0
        self.use[slot].zero_()
        self.visited[slot].zero_()
        self.path[slot] = 0.0
        self.obs_prev[slot].zero_()
        self.visited[slot, int(self.y[slot]), int(self.x[slot])] = True
        st = self.seat_stats[slot]
        st["births"] += 1
        st["gen_sum"] += gen
        st["gen_count"] += 1
        st["last_id"] = int(self.ids[slot])
        # a new body must not inherit the old body's accumulated gradient
        for t in self.brains.trainables:
            if t.grad is not None:
                t.grad[slot].zero_()
        self.births += 1
        self.log.genome(self.step_i, int(self.ids[slot]), pid, gen, float(self.energy[slot]))
        self.log.event(self.step_i, "birth", int(self.ids[slot]), pid, gen, 0,
                       float(self.energy[slot]), 0)
        self.log.seat_event(self.step_i, "birth", slot, int(self.ids[slot]), pid, gen, 0,
                            float(self.energy[slot]), 0)
        if gen > 0 and gen > self.max_gen_saved:
            self.max_gen_saved = gen
            self._save_generation(slot)

    def _die(self, slot: int) -> None:
        aid = int(self.ids[slot])
        age = int(self.age[slot])
        eaten = float(self.eaten[slot])
        self.log.event(self.step_i, "death", aid, int(self.parent[slot]), int(self.gen[slot]),
                       age, float(self.energy[slot]), eaten)
        birth = self.step_i - age
        if birth >= self.epoch_step:
            self.lifetimes.append(age)
            st = self.seat_stats[slot]
            st["deaths"] += 1
            st["total_life"] += age
            st["max_life"] = max(st["max_life"], age)
            st["total_eaten"] += eaten
            st["best_eaten"] = max(st["best_eaten"], eaten)
            if eaten > self.best_eaten:
                self.best_eaten = eaten
                self._save_hof(slot)
            if age > self.best_life:
                self.best_life = age
        self.log.seat_event(self.step_i, "death", slot, aid, int(self.parent[slot]),
                            int(self.gen[slot]), age, float(self.energy[slot]), eaten)
        self.alive[slot] = False
        self.ids[slot] = -1
        self.energy[slot] = 0.0
        self.deaths += 1

    def _save_hof(self, slot: int) -> None:
        try:
            d = Path(self.log.root) / "hall_of_fame"
            d.mkdir(parents=True, exist_ok=True)
            payload = self.brains.snapshot_slice(slot)
            payload["meta"] = {
                "id": int(self.ids[slot]),
                "gen": int(self.gen[slot]),
                "age": int(self.age[slot]),
                "eaten": float(self.eaten[slot]),
            }
            name = f"adi_{int(self.ids[slot])}_g{int(self.gen[slot])}_e{int(self.eaten[slot])}.pt"
            torch.save(payload, d / name)
        except Exception:
            pass

    def _save_generation(self, slot: int) -> None:
        try:
            d = Path(self.log.root) / "generations"
            d.mkdir(parents=True, exist_ok=True)
            payload = self.brains.snapshot_slice(slot)
            payload["meta"] = {
                "id": int(self.ids[slot]),
                "parent": int(self.parent[slot]),
                "gen": int(self.gen[slot]),
                "seat": slot,
                "step": self.step_i,
                "density": self.food_density,
                "sense": "body",
            }
            name = f"gen_{int(self.gen[slot]):04d}_id{int(self.ids[slot])}.pt"
            torch.save(payload, d / name)
        except Exception:
            pass

    def _tournament(self, k: int = 3) -> int | None:
        alive = self.alive.nonzero().flatten()
        if len(alive) == 0:
            return None
        pick = alive[torch.randint(0, len(alive), (min(k, len(alive)),))]
        best = max(pick.tolist(), key=lambda s: float(self.energy[s]))
        return int(best)

    def _reseed(self) -> None:
        free = (~self.alive).nonzero().flatten().tolist()
        for s in free[: self.cfg.init_pop]:
            self.spawn(s, None)
        self.reseeds += 1

    def set_food_density(self, target: float) -> None:
        """Change the fertile fraction of every room. Food on removed cells is lost."""
        target = max(0.0, min(0.5, float(target)))
        with torch.no_grad():
            for slot in range(self.N):
                walls = self.fields.walls[slot].numpy()
                fert = self.fields.fertility[slot].numpy()
                food = self.fields.food[slot].numpy()
                free = ~walls
                n_free = int(free.sum())
                if n_free == 0:
                    continue
                target_n = int(round(target * n_free))
                fertile = (fert > 0.5) & free
                idx = np.argwhere(fertile)
                if len(idx) > target_n:
                    drop = self.rng.choice(len(idx), size=len(idx) - target_n, replace=False)
                    for j in drop:
                        y, x = idx[j]
                        fert[y, x] = 0.0
                        food[y, x] = 0.0
                elif len(idx) < target_n:
                    empty = np.argwhere((~fertile) & free)
                    add = self.rng.choice(len(empty), size=min(target_n - len(idx), len(empty)),
                                          replace=False)
                    for j in add:
                        y, x = empty[j]
                        fert[y, x] = 1.0
                        food[y, x] = float(self.rng.uniform(0.3, 0.8))
                self.fields.fertility[slot] = torch.from_numpy(fert.astype(np.float32))
                self.fields.food[slot] = torch.from_numpy(food.astype(np.float32))
        self.food_density = target

    def cycle_target(self) -> float:
        p = self.cycle_phase / max(1.0, float(self.cycle_period))
        v = 0.5 - 0.5 * math.cos(2.0 * math.pi * p)
        return self.cycle_low + (self.cycle_high - self.cycle_low) * v

    def cycle_label(self) -> str:
        if not self.cycle_on:
            return "off"
        p = self.cycle_phase / max(1.0, float(self.cycle_period))
        if p < 0.125 or p >= 0.875:
            return "lean"
        if p < 0.375:
            return "rising"
        if p < 0.625:
            return "plenty"
        return "falling"

    def set_cycle(self, on: bool, low: float | None = None, high: float | None = None,
                  period: int | None = None, phase: float | None = None) -> None:
        self.cycle_on = bool(on)
        if low is not None:
            self.cycle_low = max(0.0, min(0.5, float(low)))
        if high is not None:
            self.cycle_high = max(self.cycle_low, min(0.5, float(high)))
        if period is not None:
            self.cycle_period = max(200, int(period))
        if phase is not None:
            self.cycle_phase = float(phase) % self.cycle_period
        if self.cycle_on:
            self.set_food_density(self.cycle_target())

    def reset_records(self) -> None:
        """Start clean life and food records from now. Older bodies do not count."""
        self.epoch_step = self.step_i
        self.best_eaten = 0.0
        self.best_life = 0
        self.lifetimes.clear()
        self.seat_stats = [self._fresh_seat_stats() for _ in range(self.N)]

    def _sotf_score(self, slot: int) -> float:
        st = self.seat_stats[slot]
        age = int(self.age[slot])
        eaten = float(self.eaten[slot])
        current = (eaten / max(1, age)) * 1000.0
        lineage = (st["total_eaten"] / max(1, st["total_life"])) * 1000.0
        births = int(st["births"])
        dead = -1000.0 if not bool(self.alive[slot]) else 0.0
        return dead + 0.6 * current + 0.3 * lineage + 0.05 * min(births, 20) + 0.001 * age

    def sotf_round(self) -> None:
        """One selection round. Top 16 keep. 17-20 stay as the experiment group.
        21-32 are replaced by children of the top 16. No reward, no teaching."""
        if not bool(self.alive.any()):
            return
        scores = [self._sotf_score(i) for i in range(self.N)]
        order = sorted(range(self.N), key=lambda i: scores[i], reverse=True)
        alive_order = [s for s in order if bool(self.alive[s])]
        top = alive_order[:16]
        near = alive_order[16:20]
        weak = alive_order[20:] + [s for s in order if not bool(self.alive[s])]
        self.sotf_session += 1
        self.sotf_last = self.step_i

        top_alive = list(top)
        parents = top_alive if top_alive else self.alive.nonzero().flatten().tolist()
        if len(parents) == 0:
            return
        weights = [max(0.01, scores[s]) for s in parents]
        total = sum(weights)
        weights = [w / total for w in weights]

        for rank, slot in enumerate(order):
            if slot in top:
                group, tier = "top", 0
            elif slot in near:
                group, tier = "near", 1
            else:
                group, tier = "weak", 2
            self.sotf_tier[slot] = tier
            st = self.seat_stats[slot]
            row = {
                "session": self.sotf_session,
                "step": self.step_i,
                "room": slot,
                "name": NAMES[slot] if slot < len(NAMES) else f"seat-{slot}",
                "group_before": group,
                "rank": rank + 1,
                "score": round(scores[slot], 4),
                "food_rate": round((float(self.eaten[slot]) / max(1, int(self.age[slot]))) * 1000.0, 4),
                "lineage_rate": round((st["total_eaten"] / max(1, st["total_life"])) * 1000.0, 4),
                "births": int(st["births"]),
                "group_after": group,
                "action": "keep",
                "parent_room": -1,
                "parent_id": -1,
                "child_id": -1,
            }
            if slot in weak:
                idx = int(self.rng.choice(len(parents), p=weights))
                parent_slot = parents[idx]
                parent_id = int(self.ids[parent_slot])
                if bool(self.alive[slot]):
                    self._die(slot)
                self.seat_stats[slot] = self._fresh_seat_stats()
                self.spawn(slot, parent_slot)
                row.update({
                    "group_after": "replaced",
                    "action": "refill",
                    "parent_room": parent_slot,
                    "parent_id": parent_id,
                    "child_id": int(self.ids[slot]),
                })
            self.log.sotf(row)
        self.log.flush()

    # ------------------------------------------------------------------ step
    def step(self) -> None:
        cfg = self.cfg
        if not bool(self.alive.any()):
            self._reseed()
            self.step_i += 1
            return

        # 1. sense
        obs = self.fields.sense(self.x, self.y, self.heading, self.energy)
        self.obs_prev = obs.detach()
        onehot = torch.nn.functional.one_hot(self.prev, cfg.act_dim).float()
        inp = torch.cat([obs, onehot], dim=1)
        h_prev = self.h.detach().clone()

        # 2. think
        h2, pred, logits = self.brains.forward(inp, self.h)
        logits = torch.clamp(logits, -20.0, 20.0)
        probs = torch.softmax(logits, dim=1)
        probs = torch.nan_to_num(probs, nan=1.0 / cfg.act_dim)
        actions = torch.multinomial(probs, 1).squeeze(1)

        # 3. act
        energy_before = self.energy.clone()
        x0 = self.x.clone()
        y0 = self.y.clone()
        self.x, self.y, self.heading, self.energy, self.eaten = self.fields.apply(
            actions, self.x, self.y, self.heading, self.energy, self.eaten)
        self.fields.regen()
        # action score: food eaten credits eat and the move that led to it
        gain = ((self.energy - energy_before) / cfg.bite).clamp(0.0, 1.0)
        got = gain > 0.01
        if bool(got.any()):
            self.action_score[self.ar, actions] += 0.10 * gain * got
            self.action_score[self.ar, self.prev] += 0.05 * gain * got
        self.action_score.mul_(0.9995).clamp_(-2.0, 4.0)
        moved = torch.sqrt((self.x - x0) ** 2 + (self.y - y0) ** 2)
        self.path = self.path + moved
        cy = self.y.long().clamp(0, cfg.grid - 1)
        cx = self.x.long().clamp(0, cfg.grid - 1)
        self.visited[self.ar, cy, cx] = True

        # use-dependent inheritance: record only use that did something. The
        # parent policy is frozen. The child gets this record at birth.
        effective = (((actions == 0) & (moved > 1e-6))
                     | (actions == 1) | (actions == 2)
                     | ((actions == 3) & got))
        with torch.no_grad():
            oh = torch.nn.functional.one_hot(actions, cfg.act_dim).float()
            self.use.mul_(self.use_decay).add_(
                (1.0 - self.use_decay) * h2.detach().unsqueeze(2) * oh.unsqueeze(1)
                * effective.float().view(-1, 1, 1))

        # 3b. energy ceiling
        at_cap = self.alive & (self.energy >= self.energy_cap - 1e-6)
        self.energy_cap = torch.where(at_cap, self.energy_cap + self.cap_growth, self.energy_cap)
        self.energy = torch.minimum(self.energy, self.energy_cap)

        # 4. burn
        self.energy = self.energy - cfg.metabolic
        self.age = self.age + self.alive.long()

        # 5. predict the next observation. this is the only learning signal.
        next_obs = self.fields.sense(self.x, self.y, self.heading, self.energy)
        err = (pred - next_obs) ** 2
        mask = self.alive.float().unsqueeze(1)
        loss = (err * mask).sum() / (mask.sum() * cfg.obs_dim + 1e-6)

        loss.backward()
        self.accum += 1
        if self.accum >= self.accum_steps:
            self.brains.adam_step()
            self.brains.zero_grad()
            self.accum = 0

        self.pred_err = err.mean(dim=1).detach()
        self.obs = next_obs.detach()
        self.probs = probs.detach()
        self.actions = actions
        self.h = h2.detach()
        self.prev = actions

        # 6. what did it notice?
        if self.step_i % 4 == 0:
            g = self.brains.salience(inp.detach(), h_prev, actions)
            groups = [
                g[:, 0:8].mean(dim=1),
                g[:, 8:9].mean(dim=1),
                g[:, 9:11].mean(dim=1),
                g[:, 11:13].mean(dim=1),
                g[:, 13:14].mean(dim=1),
                g[:, 14:18].mean(dim=1),
            ]
            s = torch.stack(groups, dim=1)
            s = s / (s.max(dim=1, keepdim=True).values + 1e-6)
            self.salience = s.detach()

        # 7. deaths
        dead = self.alive & (self.energy <= 0.0)
        if self.max_age > 0:
            dead = dead | (self.alive & (self.age >= self.max_age))
        for slot in dead.nonzero().flatten().tolist():
            self._die(slot)

        # 8. births
        free = (~self.alive).nonzero().flatten().tolist()
        eligible = (self.alive & (self.energy >= cfg.birth_energy)).nonzero().flatten().tolist()
        eligible.sort(key=lambda s: -float(self.energy[s]))
        for slot in free:
            if not eligible:
                break
            p = eligible.pop(0)
            self.energy[p] = self.energy[p] * 0.5
            self.spawn(slot, p)

        # 8b. replace a body when the room is full.
        if self.birth_mode != "off":
            while True:
                eligible = (self.alive & (self.energy >= cfg.birth_energy)).nonzero().flatten().tolist()
                if not eligible:
                    break
                p = max(eligible, key=lambda s: float(self.energy[s]))
                others = [s for s in self.alive.nonzero().flatten().tolist()
                          if s != p and int(self.age[s]) >= self.min_replace_age]
                if not others:
                    break
                if self.birth_mode == "oldest":
                    target = max(others, key=lambda s: int(self.age[s]))
                elif self.birth_mode == "threshold":
                    target = min(others, key=lambda s: float(self.energy[s]))
                    if float(self.energy[target]) >= self.birth_threshold:
                        break
                else:  # weakest
                    target = min(others, key=lambda s: float(self.energy[s]))
                self._die(target)
                self.energy[p] = self.energy[p] * 0.5
                self.spawn(target, p)

        # 9. keep the floor
        while int(self.alive.sum()) < self.min_pop:
            free = (~self.alive).nonzero().flatten().tolist()
            if not free:
                break
            p = self._tournament()
            if p is None:
                break
            self.spawn(free[0], p)

        if not bool(self.alive.any()):
            self._reseed()

        if self.cycle_on:
            self.cycle_phase = (self.cycle_phase + 1.0) % self.cycle_period
            if int(self.cycle_phase) % self.cycle_step == 0:
                self.set_food_density(self.cycle_target())

        if (self.sotf_enabled and self.sotf_interval > 0
                and (self.step_i - self.sotf_last) >= self.sotf_interval):
            self.sotf_round()

        self.step_i += 1

    # ------------------------------------------------------------ checkpoint
    def state_dict(self) -> dict:
        with torch.no_grad():
            return {
                "brains": self.brains.state_dict(),
                "walls": self.fields.walls.clone(),
                "fertility": self.fields.fertility.clone(),
                "food": self.fields.food.clone(),
                "alive": self.alive.clone(),
                "ids": self.ids.clone(),
                "parent": self.parent.clone(),
                "gen": self.gen.clone(),
                "age": self.age.clone(),
                "energy": self.energy.clone(),
                "x": self.x.clone(),
                "y": self.y.clone(),
                "heading": self.heading.clone(),
                "h": self.h.clone(),
                "prev": self.prev.clone(),
                "eaten": self.eaten.clone(),
                "pred_err": self.pred_err.clone(),
                "obs": self.obs.clone(),
                "probs": self.probs.clone(),
                "actions": self.actions.clone(),
                "salience": self.salience.clone(),
                "visited": self.visited.clone(),
                "path": self.path.clone(),
                "obs_prev": self.obs_prev.clone(),
                "next_id": self.next_id,
                "step_i": self.step_i,
                "births": self.births,
                "deaths": self.deaths,
                "reseeds": self.reseeds,
                "best_eaten": self.best_eaten,
                "best_life": self.best_life,
                "epoch_step": self.epoch_step,
                "lifetimes": list(self.lifetimes),
                "accum": self.accum,
                "rng": self.rng.bit_generator.state,
                "competitive_birth": self.competitive_birth,
                "birth_mode": self.birth_mode,
                "birth_threshold": self.birth_threshold,
                "min_replace_age": self.min_replace_age,
                "max_age": self.max_age,
                "food_density": self.food_density,
                "base_cap": self.base_cap,
                "cap_growth": self.cap_growth,
                "energy_cap": self.energy_cap.clone(),
                "action_score": self.action_score.clone(),
                "use": self.use.clone(),
                "min_pop": self.min_pop,
                "cycle_on": self.cycle_on,
                "cycle_low": self.cycle_low,
                "cycle_high": self.cycle_high,
                "cycle_period": self.cycle_period,
                "cycle_phase": self.cycle_phase,
                "sotf_enabled": self.sotf_enabled,
                "sotf_interval": self.sotf_interval,
                "sotf_last": self.sotf_last,
                "sotf_session": self.sotf_session,
                "sotf_tier": self.sotf_tier.clone(),
                "max_gen_saved": self.max_gen_saved,
                "seat_stats": [dict(s) for s in self.seat_stats],
            }

    def load_state_dict(self, d: dict) -> None:
        self.brains.load_state_dict(d["brains"])
        with torch.no_grad():
            self.fields.walls.copy_(d["walls"])
            self.fields.fertility.copy_(d["fertility"])
            self.fields.food.copy_(d["food"])
            for name in ("alive", "ids", "parent", "gen", "age", "energy",
                         "x", "y", "heading", "h", "prev", "eaten", "pred_err",
                         "obs", "probs", "actions", "salience"):
                getattr(self, name).copy_(d[name])
            if "visited" in d:
                self.visited.copy_(d["visited"])
            if "path" in d:
                self.path.copy_(d["path"])
            if "obs_prev" in d:
                self.obs_prev.copy_(d["obs_prev"])
        self.next_id = int(d["next_id"])
        self.step_i = int(d["step_i"])
        self.births = int(d["births"])
        self.deaths = int(d["deaths"])
        self.reseeds = int(d["reseeds"])
        self.best_eaten = float(d["best_eaten"])
        self.best_life = int(d.get("best_life", 0))
        self.epoch_step = int(d.get("epoch_step", 0))
        self.lifetimes = deque(d.get("lifetimes", []), maxlen=300)
        self.accum = int(d.get("accum", 0))
        self.rng.bit_generator.state = d["rng"]
        self.birth_mode = d.get("birth_mode", "threshold")
        self.birth_threshold = float(d.get("birth_threshold", 15.0))
        self.min_replace_age = int(d.get("min_replace_age", 200))
        self.max_age = int(d.get("max_age", 0))
        self.food_density = float(d.get("food_density", self.cfg.fertility))
        self.base_cap = float(d.get("base_cap", 100.0))
        self.cap_growth = float(d.get("cap_growth", 0.00001))
        if "energy_cap" in d:
            self.energy_cap.copy_(d["energy_cap"])
        else:
            self.energy_cap.fill_(self.base_cap)
        if "action_score" in d:
            self.action_score.copy_(d["action_score"])
        else:
            self.action_score.zero_()
        if "use" in d:
            self.use.copy_(d["use"])
        self.min_pop = int(d.get("min_pop", self.cfg.min_pop))
        self.cycle_on = bool(d.get("cycle_on", False))
        self.cycle_low = float(d.get("cycle_low", 0.006))
        self.cycle_high = float(d.get("cycle_high", 0.03))
        self.cycle_period = int(d.get("cycle_period", 6000))
        self.cycle_phase = float(d.get("cycle_phase", 0.0))
        self.sotf_enabled = bool(d.get("sotf_enabled", False))
        self.sotf_interval = int(d.get("sotf_interval", 0))
        self.sotf_last = int(d.get("sotf_last", 0))
        self.sotf_session = int(d.get("sotf_session", 0))
        if "sotf_tier" in d:
            self.sotf_tier.copy_(d["sotf_tier"])
        self.max_gen_saved = int(d.get("max_gen_saved", 0))
        if "seat_stats" in d:
            for i, s in enumerate(d["seat_stats"][: self.N]):
                self.seat_stats[i] = dict(s)
        # seed a baseline for seats that have a living body but no counted births
        for i in range(self.N):
            st = self.seat_stats[i]
            if bool(self.alive[i]) and int(st["births"]) == 0:
                st["births"] = 1
                st["gen_sum"] = int(self.gen[i])
                st["gen_count"] = 1
                st["last_id"] = int(self.ids[i])
        self.competitive_birth = self.birth_mode != "off"

    # ----------------------------------------------------------------- stats
    def stats(self) -> dict:
        alive = self.alive
        n = int(alive.sum())
        if n:
            birth = self.step_i - self.age[alive]
            mask = birth >= self.epoch_step
            best_live = float(self.eaten[alive][mask].max()) if bool(mask.any()) else 0.0
            best_life_live = int(self.age[alive][mask].max()) if bool(mask.any()) else 0
        else:
            best_live = 0.0
            best_life_live = 0
        return {
            "pop": n,
            "births": self.births,
            "deaths": self.deaths,
            "reseeds": self.reseeds,
            "best_eaten": max(self.best_eaten, best_live),
            "best_life": max(self.best_life, best_life_live),
            "epoch_step": self.epoch_step,
            "mean_gen": float(self.gen[alive].float().mean()) if n else 0.0,
            "mean_life": float(np.mean(self.lifetimes)) if self.lifetimes else 0.0,
            "max_gen": int(self.gen[alive].max()) if n else 0,
        }
