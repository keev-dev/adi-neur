# STATE — handoff

Last updated: 2026-10-31. This file is the restart point. Read it first.

## What this is

A small embodied agent (“Adi”) on an Oracle VM. There is no reward, no
teaching, and no hand-written controller. Bodies live in rooms, eat, die, and
are copied. Selection does the work. The current goal is a **reliable eat
reflex**. The SOTF is now live on port 8000 as an early test, with port 8001
as the control.

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
- Max age: 0 on both. Aligned for the SOTF A/B.
- SOTF: on port 8000 only. The round runs every 6,000 steps (one food
  cycle), at the start of the rising phase. The 24 new children then reach the
  plenty phase, the high food. Port 8001 is the continuous control. Groups:
  top 16 keep; ranks 17–20 stay as the experiment group; ranks 21–32 are
  replaced by children of the top 16. The top and near groups are always
  living rooms. Empty rooms always refill. Score = 0.6 current food rate +
  0.3 lineage food rate + 0.05 births + 0.001 age. No reward. See `sotf.csv`.
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

Inside a run: `events.csv`, `genome.csv`, `seat_events.csv`, `sotf.csv`,
`trace.jsonl`, `generations/`, `hall_of_fame/`, `run_meta.json`.

`sotf.csv` columns: session, step, room, name, group before, rank, score,
food rate, lineage rate, births, group after, action, parent room, parent id,
child id.

SOTF history so far, in `20261031_132511_s7/sotf.csv`:

- Session 1, step 827,970: kept empty rooms in the top 16. Grouping fixed.
- Session 2, step 833,315: filled 24 rooms from the top 8, but it ran in the
  falling phase and all 24 children died in the lean within ~3,000 steps.
- Session 3, step 847,084: ran in the rising phase. The top 8 plus 24 new
  children, population 32. The children reach the plenty phase next.
- Next automatic round: 853,084, then every 6,000 steps.

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

SOTF:

```
curl -X POST http://127.0.0.1:8000/cmd -H 'Content-Type: application/json' \
  -d '{"cmd":"sotf_now"}'
curl -X POST http://127.0.0.1:8000/cmd -H 'Content-Type: application/json' \
  -d '{"cmd":"sotf","value":1,"interval":6000}'
```

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
- SOTF children born in the falling or lean phase die within ~3,000 steps.
  Session 2 showed this. The 6,000-step schedule puts each round in the rising
  phase, so the children get the plenty phase. Watch survival.
- The first SOTF grouping kept empty rooms in the top 16. Fixed: the top and
  near groups are living rooms; empty rooms always refill.

## Decisions

- No reward, no penalty, no controller, no teaching.
- Use-dependent plasticity is allowed: it uses only the chosen action and the
  active state, not food or energy.
- The action score is a record only.
- SOTF is an early test on port 8000. It is a selection mechanism, not a
  reward. The reliable reflex is still the goal.
- The 4 near-miss rooms (ranks 17–20) are the planned experiment group.
- The witness (fixed substrate) comes later.

## Next steps

1. SOTF runs on port 8000 every 6,000 steps, in the rising phase. Port 8001
   is the control. Watch `sotf.csv`: session 3's 24 children at the plenty
   phase, whether they eat and survive the next lean, and the top 8 of
   session 4.
2. Next automatic round: 853,084, then every 6,000.
3. Measure the eat reflex at the plenty phase: `P(eat | food)` vs
   `P(eat | none)`. Target above 0.7 vs below 0.2.
4. If the reflex becomes reliable and SOTF helps, scale: trial credits can give
   8 OCPU / 16 GB for about 30 days; run 8 seeds of the same experiment.
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
| 2026-10-31 | SOTF selection raises the eat reflex | food rate of top vs bottom; reflex probe | running: session 3 at 847,084, top 8 + 24 children at rising; reflex pending | — |
