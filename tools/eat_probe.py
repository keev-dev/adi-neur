"""Eat-reflex probe. Food under the body vs no food. No learning.

Uses a saved checkpoint (all living slots) or one generation file.

    ./venv/bin/python tools/eat_probe.py --ckpt data/checkpoints/instance_1.pt
    ./venv/bin/python tools/eat_probe.py --gen data/runs/<run>/generations/gen_0001_id20.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adi.config import Config
from adi.world import Fields


def weights_from_slice(brains, slot):
    return {k: brains[k][slot].detach().clone() for k in
            ("W_ih", "b_ih", "W_hh", "b_hh", "W_pol", "b_pol")}


def weights_from_gen(d):
    return {k: d[k].detach().clone() for k in
            ("W_ih", "b_ih", "W_hh", "b_hh", "W_pol", "b_pol")}


def episode(w, cfg, seed, food, steps):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fields = Fields(cfg, 1)
    fields.new(0, rng)
    fields.fertility[0].zero_()      # no regrowth during the probe
    x, y, heading = fields.random_pos(0, rng)
    fields.food[0].zero_()
    if food:
        fields.food[0, int(y), int(x)] = 1.0

    xt = torch.tensor([x], dtype=torch.float32)
    yt = torch.tensor([y], dtype=torch.float32)
    ht = torch.tensor([heading], dtype=torch.float32)
    en = torch.tensor([cfg.start_energy], dtype=torch.float32)
    eaten_t = torch.zeros(1)
    h = torch.zeros(1, cfg.hidden)
    prev = 0
    H = cfg.hidden

    eat_first = 0
    eat_any = 0
    for t in range(steps):
        obs = fields.sense(xt, yt, ht, en)
        onehot = F.one_hot(torch.tensor([prev]), cfg.act_dim).float()
        inp = torch.cat([obs, onehot], dim=1)
        gi = inp @ w["W_ih"] + w["b_ih"]
        gh = h @ w["W_hh"] + w["b_hh"]
        r = torch.sigmoid(gi[:, :H] + gh[:, :H])
        z = torch.sigmoid(gi[:, H:2 * H] + gh[:, 2 * H:])
        n = torch.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h = (1.0 - z) * n + z * h
        logits = h @ w["W_pol"] + w["b_pol"]
        probs = torch.softmax(logits, dim=1)
        a = int(torch.multinomial(probs, 1).item())
        if t == 0 and a == 3:
            eat_first = 1
        if a == 3:
            eat_any = 1
        prev = a
        xt, yt, ht, en, eaten_t = fields.apply(
            torch.tensor([a]), xt, yt, ht, en, eaten_t)
        en = en - cfg.metabolic
        if float(en) <= 0:
            break

    return eat_first, eat_any, int(eaten_t)


def probe(name, w, cfg, seeds, steps):
    first_food = first_none = any_food = any_none = 0
    eat_food = eat_none = 0
    for s in range(seeds):
        a, b, c = episode(w, cfg, s, True, steps)
        first_food += a; any_food += b; eat_food += c
        a, b, c = episode(w, cfg, s, False, steps)
        first_none += a; any_none += b; eat_none += c
    n = seeds
    print(f"{name:<18} "
          f"eat@food={first_food/n:.2f} eat@none={first_none/n:.2f} | "
          f"any@food={any_food/n:.2f} any@none={any_none/n:.2f} | "
          f"cells food={eat_food/n:.2f} none={eat_none/n:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--gen", default=None)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--slot", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    torch.set_num_threads(1)

    if args.gen:
        d = torch.load(args.gen, map_location="cpu", weights_only=False)
        meta = d.get("meta", {})
        print("generation", meta)
        probe(f"gen {meta.get('gen')} id {meta.get('id')}", weights_from_gen(d),
              cfg, args.seeds, args.steps)
        return

    path = args.ckpt or "data/checkpoints/instance_1.pt"
    d = torch.load(path, map_location="cpu", weights_only=False)
    pop = d["pop"]
    brains = pop["brains"]
    alive = pop["alive"]
    ids = pop["ids"]
    gen = pop["gen"]
    slots = [args.slot] if args.slot is not None else [
        i for i in range(len(alive)) if bool(alive[i])]
    print("checkpoint", path, "alive", len(slots), "steps", args.steps, "seeds", args.seeds)
    for slot in slots:
        probe(f"slot {slot} id {int(ids[slot])} g{int(gen[slot])}",
              weights_from_slice(brains, slot), cfg, args.seeds, args.steps)


if __name__ == "__main__":
    main()
