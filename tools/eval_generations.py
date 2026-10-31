"""Evaluate saved generations on fresh rooms with no learning.

The brain is frozen. The body acts for N steps in a new room. This measures the
inherited behaviour, without luck from a long life and without lifetime learning.

Usage:
    python tools/eval_generations.py --run data/runs/<run> --steps 1500 --seeds 5 --every 5
"""

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adi.config import Config
from adi.world import Fields


def evaluate(brain, cfg, seed, steps):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fields = Fields(cfg, 1)
    fields.new(0, rng)
    x, y, heading = fields.random_pos(0, rng)

    xt = torch.tensor([x], dtype=torch.float32)
    yt = torch.tensor([y], dtype=torch.float32)
    ht = torch.tensor([heading], dtype=torch.float32)
    en = torch.tensor([cfg.start_energy], dtype=torch.float32)
    eaten_t = torch.zeros(1)
    h = torch.zeros(1, cfg.hidden)
    prev = 0
    visited = torch.zeros(1, cfg.grid, cfg.grid, dtype=torch.bool)
    visited[0, int(y), int(x)] = True
    path = 0.0
    H = cfg.hidden

    W_ih, b_ih = brain["W_ih"], brain["b_ih"]
    W_hh, b_hh = brain["W_hh"], brain["b_hh"]
    W_pol, b_pol = brain["W_pol"], brain["b_pol"]

    done = 0
    for done in range(1, steps + 1):
        obs = fields.sense(xt, yt, ht, en)
        onehot = torch.nn.functional.one_hot(torch.tensor([prev]), cfg.act_dim).float()
        inp = torch.cat([obs, onehot], dim=1)

        gi = inp @ W_ih + b_ih
        gh = h @ W_hh + b_hh
        r = torch.sigmoid(gi[:, :H] + gh[:, :H])
        z = torch.sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = torch.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h = (1.0 - z) * n + z * h

        logits = h @ W_pol + b_pol
        probs = torch.softmax(logits, dim=1)
        a = int(torch.multinomial(probs, 1).item())
        prev = a

        x0, y0 = xt.clone(), yt.clone()
        xt, yt, ht, en, eaten_t = fields.apply(
            torch.tensor([a]), xt, yt, ht, en, eaten_t)
        fields.regen()
        en = en - cfg.metabolic

        path += float(torch.sqrt((xt - x0) ** 2 + (yt - y0) ** 2))
        visited[0, int(yt), int(xt)] = True
        if float(en) <= 0.0:
            break

    return {
        "eaten": int(eaten_t),
        "steps": done,
        "coverage": float(visited.float().mean()),
        "path": path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.run:
        run = Path(args.run)
    else:
        runs = [p for p in (ROOT / "data" / "runs").iterdir() if p.is_dir()]
        run = sorted(runs)[-1]

    files = sorted((run / "generations").glob("gen_*.pt"))
    if not files:
        print("no generation files in", run / "generations")
        return
    files = files[::max(1, args.every)]

    base_cfg = Config()
    torch.set_num_threads(1)
    rows = []
    print(f"{'gen':>5} {'n':>3} {'eaten':>7} {'steps':>7} {'cover':>7} {'path':>8}")
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        gen = int(d.get("meta", {}).get("gen", -1))
        density = (args.density if args.density is not None
                   else float(d.get("meta", {}).get("density", base_cfg.fertility)))
        cfg = dataclasses.replace(base_cfg, fertility=density)
        res = [evaluate(d, cfg, s, args.steps) for s in range(args.seeds)]
        n = len(res)
        row = {
            "gen": gen,
            "n": n,
            "density": density,
            "eaten": sum(r["eaten"] for r in res) / n,
            "steps": sum(r["steps"] for r in res) / n,
            "coverage": sum(r["coverage"] for r in res) / n,
            "path": sum(r["path"] for r in res) / n,
        }
        rows.append(row)
        print(f"{gen:5d} {n:3d} {row['eaten']:7.2f} {row['steps']:7.0f} "
              f"{row['coverage'] * 100:6.1f}% {row['path']:8.1f}")

    out = Path(args.out) if args.out else run / "eval_generations.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
