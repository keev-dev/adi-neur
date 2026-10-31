# STATE — handoff

Last updated: 2026-10-31. This file is the restart point. Read it first.

## What this is

A small embodied agent (“Adi”) on an Oracle VM. There is no reward, no
teaching, and no hand-written controller. Bodies live in rooms, eat, die, and
are copied. Selection does the work. The current goal is a **reliable eat
reflex** before any SOTF rounds or scaling.

## Where it runs

- VM: `129.159.156.155`, user `ubuntu`, key `C:\Users\Harsh\.ssh\id_ed25519`.
- Repo: https://github.com/keev-dev/adi-neur (public).
- Local project: `E:\neurExp`.
- Services: `adi@1` (port 8000, seed 7), `adi@2` (port 8001, seed 8).
- Both bind `127.0.0.1`. Only SSH is public.
- VM: Oracle A1, 2 OCPU, 12 GB, Ubuntu 24.04, `/opt/adi`.

## URLs

Local tunnel (run on the laptop):

```
ssh -N -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 -i "C:\Users\Harsh\.ssh\id_ed25519" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 ubuntu@129.159.156.155
```

- http://127.0.0.1:8000
- http://127.0.0.1:8001

Cloudflare quick tunnels (public, temporary):

- https://chef-mba-friendship-seminars.trycloudflare.com (8000)
- https://standard-trained-brown-sees.trycloudflare.com (8001)

These change if `cloudflared` restarts. Do not share them. To restart:

```
mkdir -p /opt/adi/data/tunnels
setsid nohup cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate \
  > /opt/adi/data/tunnels/cf8000.log 2>&1 < /dev/null &
setsid nohup cloudflared tunnel --url http://127.0.0.1:8001 --no-autoupdate \
  > /opt/adi/data/tunnels/cf8001.log 2>&1 < /dev/null &
grep -o 'https://[a-zA-Z0-9.-]*trycloudflare.com' /opt/adi/data/tunnels/cf8000.log | head -1
grep -o 'https://[a-zA-Z0-9.-]*trycloudflare.com' /opt/adi/data/tunnels/cf8001.log | head -1
```

## Current architecture

- 14 senses: 8 whiskers, food under body, food ahead, food left (body frame),
  heading sin/cos, energy.
- 4 actions: forward, left, right, eat.
- Memory: GRU, 128 hidden units. Input: 18 (14 senses + 4 previous action).
- Prediction head: 128 → 14. Learns online to predict the next senses.
- Action head: 128 → 4. **Use-dependent plasticity** (new): the chosen
  action’s connections to the active memory cells grow by `0.001 * h` each
  step. All action weights decay by `0.9995`; clamped to `[-2, 2]`.
  No reward. The child inherits these weights.
- Action score: 4 numbers, updated when food is eaten; **record only**, not a
  bias (strict mode). The child inherits half.
- No action-score bias in the logits. No blocked penalty. No controller.
- Child: copies the parent’s current brain (memory, prediction, action head),
  then mutation. Action mutation: 2% of weights, sigma 0.05. Body mutation:
  0.2%, sigma 0.02.

## Current rules

- Food cycle: low 0.006 (≈12 cells), high 0.03 (≈60 cells), period 6,000
  steps, updated every 100 steps. Port 8000 and 8001 have opposite phases.
- Energy cap: 100. The cap grows by 0.00001 per step at the cap.
- Floor: `min_pop` 8.
- Max age: 8,000 on port 8000; off on port 8001.
- Birth rule: `threshold` (replace weakest below 15 energy only).
- Records reset epoch: 726,084 on port 8000; 756,611 on port 8001. Only bodies
  born after these steps count for `best eat` / `best life` and the per-seat
  means. The old 88 / 355 records are intentionally discarded.
- Strict mode: no reward, no penalty, no teaching. Selection only.

## Data

On the VM:

```
/opt/adi/data/runs/20261031_132511_s7/   # port 8000
/opt/adi/data/runs/20261031_132511_s8/   # port 8001
/opt/adi/data/checkpoints/instance_1.pt
/opt/adi/data/checkpoints/instance_2.pt
/opt/adi/data/tunnels/cf8000.log
/opt/adi/data/tunnels/cf8001.log
```

Inside a run: `events.csv`, `genome.csv`, `seat_events.csv`, `trace.jsonl`,
`generations/`, `hall_of_fame/`, `run_meta.json`.

Trace rows carry `sense: body` after the body-frame change. Rows without it
use the old world-frame senses. Do not mix them in one analysis.

## Commands

Status (through the local tunnel):

```
python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/state'))['stats'])"
```

Operate:

```
ssh -i "C:\Users\Harsh\.ssh\id_ed25519" ubuntu@129.159.156.155
sudo systemctl status adi@1 adi@2
sudo systemctl restart adi@1 adi@2
journalctl -u adi@1 -n 50 --no-pager
```

Safe deploy (pause → checkpoint → pull → restart → resume):

```
ssh -i "C:\Users\Harsh\.ssh\id_ed25519" ubuntu@129.159.156.155 "cd /opt/adi && bash deploy/pull.sh --restart"
```

Without `--restart`, the script checkpoints and pulls but leaves the services
paused. Use `--restart` for a normal update.

Local test (port 8010 avoids the tunnel):

```
python run.py --instance 1 --port 8010 --fresh
```

## Evaluation

`tools/eval_generations.py` runs saved generation brains in fresh rooms with
no learning. Use it only on generation files with `meta.sense == "body"`.
Example:

```
cd /opt/adi
./venv/bin/python tools/eval_generations.py \
  --run data/runs/20261031_132511_s7 \
  --steps 1000 --seeds 5 --every 5 --density 0.006 --cap 100
```

The eat probe (`tools/eat_probe.py`) tests the reflex: food under the body vs
no food, using a saved checkpoint. Run it on the VM after the use-dependent
head has been live for a while.

## Known issues

- The local SSH tunnel dies during long restarts or network changes. Restart
  the command above. The Cloudflare links do not need the laptop tunnel.
- Cloudflare quick URLs are public and temporary. Do not share them.
- The world model is passive: it predicts the next senses, not the effect of
  each action. The action head is now plastic, but there is still no planning.
- Arjuna (port 8001) died stuck on a wall while spamming forward and eat. This
  is the reflex ceiling, not a bug. Selection removes such lines.
- Pre-body-frame generation files are not valid for the current eval.
- The old best records (88 / 355) are discarded by the epoch reset.

## Decisions

- No reward, no penalty, no controller, no teaching.
- Use-dependent plasticity is allowed: it uses only the chosen action and the
  active state, not food or energy.
- The action score is a record only.
- SOTF rounds are postponed until the eat reflex is reliable.
- The 4 near-miss rooms (ranks 17–20) are the planned experiment group.
- The witness (fixed substrate) comes later.

## Next steps

1. Let the use-dependent action head run. Measure `P(eat | food)` vs
   `P(eat | none)` from the trace and the eat probe. Target: a clear gap, e.g.
   above 0.7 vs below 0.2.
2. If the reflex becomes reliable, run SOTF on one instance:
   - every 20,000 steps,
   - top 16 keep, ranks 17–20 experiment, ranks 21–32 refill from top 16,
   - score: food rate first, births second, age tiebreak,
   - CSV: session, step, room, group before/after, rank, food rate, parent,
     child.
3. Keep the other instance continuous as the control.
4. Only then scale: trial credits can give 8 OCPU / 16 GB for about 30 days;
   run 8 seeds of the same experiment, one per core.
5. Then the action-conditioned (what-if) model, only if the reflex is stable.

## Outside research to compare

Melting Pot, EvoJAX/evosax, DreamerV3, Lenia/Avida, MAP-Elites/POET/ASAL,
homeostatic RL, Butlin et al. 2023 (consciousness indicators). The witness and
observer ablation still look novel.

## Ledger

| Date | Hypothesis | Metric | Result | Verdict |
|---|---|---|---|---|
| 2026-10-31 | Food rate beats age as a selection score | cells per 1,000 steps | pending | — |
| 2026-10-31 | Use-dependent action head creates a reliable eat reflex | P(eat \| food) vs P(eat \| none) | pending | — |
