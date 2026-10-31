"""The living loop, the focused line, and the snapshot the browser sees."""

import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import torch

from .config import Config
from .log import Logger
from .names import NAMES, hue_for, vitality
from .population import Population

ROOT = Path(__file__).resolve().parent.parent


class Simulation:
    def __init__(self, cfg, root: Path | None = None,
                 checkpoint_path: Path | None = None, fresh: bool = False):
        self.cfg = cfg
        self.root = Path(root) if root else ROOT
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        self.lock = threading.RLock()
        self.running = False
        self.paused = False
        self.speed = 1.0          # 0.0 means as fast as possible
        self._thread: threading.Thread | None = None

        self.focus_id: int | None = None
        self.focus_slot: int | None = None
        self.focus_mode = "auto"   # auto | manual
        self.trail = deque(maxlen=140)
        self.map_pts = deque(maxlen=360)
        self.err_hist = deque(maxlen=240)

        # fixed random projection of the recurrent state. stable across a run.
        g = torch.Generator().manual_seed(1234)
        self.proj = torch.randn(cfg.hidden, 2, generator=g) / math.sqrt(cfg.hidden)

        self._wall_cache_id = None
        self._wall_cache = []
        self._last_pub = 0.0
        self._last_ckpt = time.perf_counter()
        self.latest: dict | None = None
        self.resumed = False

        ckpt = None
        if self.checkpoint_path and self.checkpoint_path.exists() and not fresh:
            try:
                ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            except Exception:
                ckpt = None

        if ckpt is not None and self._compatible(ckpt):
            stored = ckpt.get("config", {})
            if stored:
                try:
                    self.cfg = Config(**{k: v for k, v in stored.items()
                                         if k in Config.__dataclass_fields__})
                except Exception:
                    pass
            self.resumed = True
            self.run_dir = Path(ckpt["run_dir"])
            self.run_id = ckpt.get("run_id", self.run_dir.name)
            self.log = Logger(self.run_dir)
            torch.manual_seed(self.cfg.seed)
            self.pop = Population(self.cfg, self.cfg.seed, self.log, init=False)
            self._restore(ckpt)
            self.latest = self._build_snapshot()
            print(f"resumed {self.run_id} at step {self.pop.step_i} from {self.checkpoint_path}")
        else:
            run_id = time.strftime("%Y%m%d_%H%M%S") + f"_s{cfg.seed}"
            self.run_dir = self.root / "data" / "runs" / run_id
            self.run_id = run_id
            self.log = Logger(self.run_dir)
            with open(self.run_dir / "run_meta.json", "w", encoding="utf-8") as f:
                json.dump(asdict(cfg), f, indent=2)
            torch.manual_seed(cfg.seed)
            self.pop = Population(cfg, cfg.seed, self.log)
            self._refocus()
            self.latest = self._build_snapshot()

    # ------------------------------------------------------------- checkpoint
    def _compatible(self, ckpt: dict) -> bool:
        if int(ckpt.get("version", 0)) != 1:
            return False
        c = ckpt.get("config", {})
        return (
            c.get("n_slots") == self.cfg.n_slots
            and c.get("hidden") == self.cfg.hidden
            and c.get("obs_dim") == self.cfg.obs_dim
            and c.get("act_dim") == self.cfg.act_dim
            and c.get("grid") == self.cfg.grid
        )

    def save_checkpoint(self, path: Path | str | None = None) -> bool:
        p = Path(path) if path is not None else self.checkpoint_path
        if p is None:
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            data = {
                "version": 1,
                "config": asdict(self.cfg),
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "pop": self.pop.state_dict(),
                "focus_id": self.focus_id,
                "focus_slot": self.focus_slot,
                "focus_mode": self.focus_mode,
                "trail": list(self.trail),
                "map_pts": list(self.map_pts),
                "err_hist": list(self.err_hist),
                "speed": self.speed,
                "paused": self.paused,
                "torch_rng": torch.get_rng_state(),
            }
            tmp = p.with_name(p.name + ".tmp")
            torch.save(data, tmp)
            os.replace(tmp, p)
        self._last_ckpt = time.perf_counter()
        return True

    def _restore(self, ckpt: dict) -> None:
        with self.lock:
            self.pop.load_state_dict(ckpt["pop"])
            self.focus_id = ckpt.get("focus_id")
            self.focus_slot = ckpt.get("focus_slot")
            self.focus_mode = ckpt.get("focus_mode", "auto")
            self.trail = deque(ckpt.get("trail", []), maxlen=140)
            self.map_pts = deque(ckpt.get("map_pts", []), maxlen=900)
            self.err_hist = deque(ckpt.get("err_hist", []), maxlen=240)
            self.speed = float(ckpt.get("speed", 1.0))
            self.paused = bool(ckpt.get("paused", False))
            if "torch_rng" in ckpt:
                torch.set_rng_state(ckpt["torch_rng"])
            self._last_ckpt = time.perf_counter()
            if self.focus_mode == "manual" and self.focus_slot is not None:
                if bool(self.pop.alive[self.focus_slot]):
                    self.focus_id = int(self.pop.ids[self.focus_slot])
                else:
                    self.focus_id = None
            else:
                self.focus_mode = "auto"
            if self._focus_slot() is None:
                self._refocus()

    # ----------------------------------------------------------------- focus
    def _refocus(self) -> None:
        if self.focus_mode == "manual" and self.focus_slot is not None:
            if bool(self.pop.alive[self.focus_slot]):
                self.focus_id = int(self.pop.ids[self.focus_slot])
            else:
                self.focus_id = None
            return

        alive = self.pop.alive.nonzero().flatten().tolist()
        if not alive:
            self.focus_id = None
            return
        best = max(alive, key=lambda s: (int(self.pop.gen[s]), float(self.pop.energy[s])))
        self.focus_id = int(self.pop.ids[best])
        self.focus_slot = best
        self.trail.clear()
        self.map_pts.clear()
        self.err_hist.clear()

    def _focus_slot(self) -> int | None:
        if self.focus_mode == "manual" and self.focus_slot is not None:
            i = self.focus_slot
            if 0 <= i < self.pop.N and bool(self.pop.alive[i]):
                new_id = int(self.pop.ids[i])
                if self.focus_id is not None and new_id != self.focus_id:
                    self.trail.clear()
                    self.map_pts.clear()
                    self.err_hist.clear()
                self.focus_id = new_id
                return i
            return None
        if self.focus_id is None:
            return None
        hit = (self.pop.ids == self.focus_id).nonzero().flatten()
        return int(hit[0]) if len(hit) else None

    # ------------------------------------------------------------------- step
    def step_once(self) -> None:
        with self.lock:
            self.pop.step()
            f = self._focus_slot()
            if f is None:
                if self.focus_mode != "manual":
                    self._refocus()
                    f = self._focus_slot()
                if f is None:
                    now = time.perf_counter()
                    if now - self._last_pub >= 1.0 / self.cfg.snap_hz:
                        self.latest = self._build_snapshot()
                        self._last_pub = now
                    if self.checkpoint_path and now - self._last_ckpt >= 600.0:
                        self.save_checkpoint()
                    return

            self.trail.append((float(self.pop.x[f]), float(self.pop.y[f])))

            h = self.pop.h[f].detach()
            p = (h @ self.proj).tolist()
            self.map_pts.append((p[0], p[1], self._label(f)))
            self.err_hist.append(float(self.pop.pred_err[f]))

            if self.pop.step_i % 20 == 0:
                self.log.trace({
                    "step": self.pop.step_i,
                    "seat": f,
                    "name": NAMES[f] if f < len(NAMES) else f"seat-{f}",
                    "id": int(self.pop.ids[f]),
                    "gen": int(self.pop.gen[f]),
                    "age": int(self.pop.age[f]),
                    "energy": round(float(self.pop.energy[f]), 3),
                    "action": int(self.pop.actions[f]),
                    "pred_err": round(float(self.pop.pred_err[f]), 6),
                    "coverage": round(float(self.pop.visited[f].float().mean()), 4),
                    "path": round(float(self.pop.path[f]), 2),
                    "sense": "body",
                    "sal": [round(float(v), 5) for v in self.pop.salience[f]],
                    "h": [round(float(v), 4) for v in self.pop.h[f]],
                    "obs": [round(float(v), 4) for v in self.pop.obs_prev[f]],
                })
                if self.pop.step_i % 200 == 0:
                    self.log.flush()

            now = time.perf_counter()
            if now - self._last_pub >= 1.0 / self.cfg.snap_hz:
                self.latest = self._build_snapshot()
                self._last_pub = now

            if self.checkpoint_path and now - self._last_ckpt >= 600.0:
                self.save_checkpoint()

    def _label(self, f: int) -> int:
        if float(self.pop.energy[f]) < 20.0:
            return 3
        obs = self.pop.obs[f]
        if float(obs[0:8].min()) > 0.75:
            return 2
        if float(obs[8]) > 0.05:
            return 1
        return 0

    # ------------------------------------------------------------------- loop
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="adi-sim")
        self._thread.start()

    def _loop(self) -> None:
        n = 0
        while self.running:
            if self.paused:
                time.sleep(0.01)
                continue
            t0 = time.perf_counter()
            self.step_once()
            if self.speed > 0.0:
                target = 1.0 / (self.cfg.base_hz * self.speed)
                dt = time.perf_counter() - t0
                if dt < target:
                    time.sleep(target - dt)
            else:
                n += 1
                if n % 16 == 0:
                    time.sleep(0)

    def stop(self) -> None:
        """Stop the loop, save a checkpoint, flush the logs."""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        try:
            self.save_checkpoint()
        except Exception:
            pass
        try:
            self.log.flush()
            self.log.close()
        except Exception:
            pass

    def command(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        with self.lock:
            if cmd == "speed":
                v = float(msg.get("value", 1))
                self.speed = 0.0 if v <= 0 else v
            elif cmd == "pause":
                self.paused = not self.paused
            elif cmd == "set_paused":
                self.paused = bool(msg.get("value", True))
            elif cmd == "resume":
                self.paused = False
            elif cmd == "checkpoint":
                self.save_checkpoint()
            elif cmd == "churn":
                on = bool(msg.get("value", True))
                self.pop.birth_mode = "threshold" if on else "off"
                self.pop.competitive_birth = on
            elif cmd == "birth_mode":
                mode = msg.get("mode") or "threshold"
                if mode in ("weakest", "threshold", "oldest", "off"):
                    self.pop.birth_mode = mode
                    self.pop.competitive_birth = mode != "off"
            elif cmd == "max_age":
                self.pop.max_age = max(0, int(msg.get("value", 0)))
            elif cmd == "food_density":
                self.pop.cycle_on = False
                self.pop.set_food_density(float(msg.get("value", self.cfg.fertility)))
            elif cmd == "cycle":
                self.pop.set_cycle(bool(msg.get("value", True)),
                                   msg.get("low"), msg.get("high"),
                                   msg.get("period"), msg.get("phase"))
            elif cmd == "min_pop":
                self.pop.min_pop = max(1, int(msg.get("value", 8)))
            elif cmd == "energy_cap":
                v = float(msg.get("value", 100.0))
                self.pop.base_cap = v
                self.pop.energy_cap.fill_(v)
                self.pop.energy = torch.minimum(self.pop.energy, self.pop.energy_cap)
            elif cmd == "focus":
                if msg.get("mode") == "auto":
                    self.focus_mode = "auto"
                    self.focus_slot = None
                    self._refocus()
                else:
                    slot = int(msg.get("slot", -1))
                    if 0 <= slot < self.pop.N:
                        self.focus_mode = "manual"
                        self.focus_slot = slot
                        self.focus_id = (int(self.pop.ids[slot])
                                         if bool(self.pop.alive[slot]) else None)
                        self.trail.clear()
                        self.map_pts.clear()
                        self.err_hist.clear()

    # --------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        """The latest published frame. O(1), safe to read from another thread."""
        if self.latest is None:
            self.latest = self._build_snapshot()
        else:
            self.latest["paused"] = self.paused
            self.latest["speed"] = self.speed
            self.latest["birth_mode"] = self.pop.birth_mode
            self.latest["churn"] = self.pop.birth_mode != "off"
            self.latest["max_age"] = self.pop.max_age
            self.latest["food_density"] = round(float(self.pop.food_density), 4)
            self.latest["energy_cap"] = round(float(self.pop.base_cap), 4)
            self.latest["min_pop"] = self.pop.min_pop
            self.latest["cycle_on"] = self.pop.cycle_on
            self.latest["cycle_label"] = self.pop.cycle_label()
            self.latest["cycle_phase"] = round(float(self.pop.cycle_phase), 1)
        return self.latest

    def _build_snapshot(self) -> dict:
        with self.lock:
            f = self._focus_slot()
            stats = self.pop.stats()

            roster = []
            for i in range(self.pop.N):
                alive = bool(self.pop.alive[i])
                en = float(self.pop.energy[i])
                ea = float(self.pop.eaten[i])
                ag = int(self.pop.age[i])
                gn = int(self.pop.gen[i])
                roster.append({
                    "slot": i,
                    "name": NAMES[i] if i < len(NAMES) else f"seat-{i}",
                    "hue": round(hue_for(i, self.pop.N), 1),
                    "alive": alive,
                    "id": int(self.pop.ids[i]),
                    "gen": gn,
                    "age": ag,
                    "energy": round(max(0.0, en), 1),
                    "cap": round(float(self.pop.energy_cap[i]), 6),
                    "eaten": int(ea),
                    "rate": round(ea / max(1, ag) * 1000.0, 2),
                    "coverage": round(float(self.pop.visited[i].float().mean()), 4),
                    "path": round(float(self.pop.path[i]), 1),
                    "probs": [round(float(v), 3) for v in self.pop.probs[i]],
                    "action": int(self.pop.actions[i]),
                    "pred_err": round(float(self.pop.pred_err[i]), 5),
                    "vitality": round(vitality(en, ea, ag, gn) if alive else 0.0, 3),
                    "focused": i == f,
                })

            seat_stats = []
            for i in range(self.pop.N):
                st = self.pop.seat_stats[i]
                seat_stats.append({
                    "slot": i,
                    "name": NAMES[i] if i < len(NAMES) else f"seat-{i}",
                    "births": int(st["births"]),
                    "deaths": int(st["deaths"]),
                    "mean_life": round(st["total_life"] / max(1, st["deaths"]), 1),
                    "max_life": int(st["max_life"]),
                    "mean_gen": round(st["gen_sum"] / max(1, st["gen_count"]), 2),
                    "mean_food": round(st["total_eaten"] / max(1, st["deaths"]), 2),
                    "best_food": round(st["best_eaten"], 1),
                    "current_id": int(self.pop.ids[i]),
                    "current_gen": int(self.pop.gen[i]),
                    "current_age": int(self.pop.age[i]),
                    "current_energy": round(max(0.0, float(self.pop.energy[i])), 1),
                    "alive": bool(self.pop.alive[i]),
                })

            base = {
                "step": self.pop.step_i,
                "speed": self.speed,
                "paused": self.paused,
                "focus_id": self.focus_id,
                "focus_slot": self.focus_slot,
                "focus_mode": self.focus_mode,
                "churn": self.pop.birth_mode != "off",
                "birth_mode": self.pop.birth_mode,
                "max_age": self.pop.max_age,
                "food_density": round(float(self.pop.food_density), 4),
                "energy_cap": round(float(self.pop.base_cap), 4),
                "min_pop": self.pop.min_pop,
                "cycle_on": self.pop.cycle_on,
                "cycle_low": round(float(self.pop.cycle_low), 4),
                "cycle_high": round(float(self.pop.cycle_high), 4),
                "cycle_period": self.pop.cycle_period,
                "cycle_phase": round(float(self.pop.cycle_phase), 1),
                "cycle_label": self.pop.cycle_label(),
                "stats": stats,
                "roster": roster,
                "seat_stats": seat_stats,
                "map": [[round(x, 3), round(y, 3), l] for (x, y, l) in self.map_pts],
                "trail": [[round(x, 2), round(y, 2)] for (x, y) in self.trail],
                "err_hist": [round(v, 5) for v in self.err_hist],
            }
            if f is None:
                base["agent"] = None
                base["world"] = None
                return base

            G = self.cfg.grid
            if self._wall_cache_id != int(self.pop.ids[f]):
                nz = self.pop.fields.walls[f].nonzero()
                self._wall_cache = [[int(x), int(y)] for y, x in nz.tolist()]
                self._wall_cache_id = int(self.pop.ids[f])
            food_t = self.pop.fields.food[f]
            nzf = (food_t > 0.02).nonzero()
            food = [[int(x), int(y), round(float(food_t[y, x]), 2)] for y, x in nzf.tolist()]

            obs = self.pop.obs[f]
            probs = self.pop.probs[f]
            en = float(self.pop.energy[f])
            ea = float(self.pop.eaten[f])
            ag = int(self.pop.age[f])
            gn = int(self.pop.gen[f])
            base["world"] = {
                "size": G,
                "range": self.cfg.max_range,
                "walls": self._wall_cache,
                "food": food,
            }
            base["agent"] = {
                "id": int(self.pop.ids[f]),
                "slot": f,
                "name": NAMES[f] if f < len(NAMES) else f"seat-{f}",
                "hue": round(hue_for(f, self.pop.N), 1),
                "vitality": round(vitality(en, ea, ag, gn), 3),
                "parent": int(self.pop.parent[f]),
                "gen": gn,
                "age": ag,
                "energy": round(en, 2),
                "cap": round(float(self.pop.energy_cap[f]), 6),
                "x": round(float(self.pop.x[f]), 2),
                "y": round(float(self.pop.y[f]), 2),
                "heading": round(float(self.pop.heading[f]), 3),
                "whiskers": [round(float(v), 3) for v in obs[0:8]],
                "food_here": round(float(obs[8]), 3),
                "action": int(self.pop.actions[f]),
                "probs": [round(float(v), 4) for v in probs],
                "pred_err": float(self.pop.pred_err[f]),
                "eaten": int(ea),
                "coverage": round(float(self.pop.visited[f].float().mean()), 4),
                "path": round(float(self.pop.path[f]), 1),
                "units": [round(float(v), 3) for v in self.pop.h[f]],
                "sal": [round(float(v), 3) for v in self.pop.salience[f]],
            }
            return base
