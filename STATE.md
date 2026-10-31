# STATE — handoff

Last updated: 2026-10-31. This file is the restart point. Read it first.

## What this is

A small embodied agent (“Adi”) on an Oracle VM. There is no reward, no
teaching, and no hand-written controller. Bodies live in rooms, eat, die, and
are copied. Selection does the work. The current goal is a **reliable eat
reflex**. SOTF is off after a failed early test on port 8000: the 24 children
died each round at pop 8. Port 8001 is the control. On 2026-10-31 the action
head moved to use-dependent inheritance at birth, and the score boards were
reset so the next records belong to this mechanism. The post-reset population
swings from 8 in the lean phase to 32 in the plenty phase.

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
- Action head: 128 → 4. **Use-dependent inheritance**: the parent’s action
  head is fixed for life. The parent keeps a decayed average of *effective*
  use only: forward that moved, turn that changed heading, eat that got food.
  At birth the child’s head = parent’s head + `2.0 × use record`, then normal
  mutation. Clamped to `[-2, 2]`. The 2.0 matches the old 0.001-per-step total
  over the ~2,000-step decay horizon. No reward. The child’s own use starts
  empty.
- Action score: 4 numbers, updated when food is eaten; **record only**, not a
  bias (strict mode). The child inherits half.
- No action-score bias in the logits. No blocked penalty. No controller.
- Child: copies the parent’s brain (memory, prediction, action head), adds the
  parent’s effective-use boost to the action head, then mutates. Action
  mutation: 2% of weights, sigma 0.05. Body mutation: 0.2%, sigma 0.02.

## Current rules

- Food cycle: low 0.006 (≈12 cells), high 0.03 (≈60 cells), period 6,000
  steps, updated every 100 steps. Port 8000 and 8001 have opposite phases.
- Energy cap: 100. The cap grows by 0.00001 per step at the cap.
- Floor: `min_pop` 8.
- Max age: 0 on both. Aligned for the SOTF A/B.
- SOTF: **off on port 8000** after 5 sessions. Each round added 24 children;
  all 24 died and the population returned to 8. The code and `sotf.csv`
  remain. The food/energy is phase-dependent: the lean phase falls to ~8, the
  plenty phase can fill 32 (8001 hit 32 on 2026-10-31). Do not turn SOTF on
  until the phase/floor policy is chosen. Port 8001 is the continuous control.
- Birth rule: `threshold` (replace weakest below 15 energy only).
- Records reset epoch: 920,133 on port 8000; 943,206 on port 8001 (reset
  2026-10-31 after the use-inheritance deploy). Only bodies born after these
  steps count for `best eat` / `best life` and the per-seat means. The old
  88 / 355 records are intentionally discarded.
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
- Session 5, step 859,084: 24 children; all died; population 8. SOTF off.
- SOTF is now off on 8000. The use-inheritance code is live on both. After the
  2026-10-31 reset the population swings with the food cycle: at step ~924.6k
  8000 fell to 8 (lean) while at step ~947.7k 8001 reached 32 (all seats,
  plenty). The reset-run records at those steps: best eat 11 / 15, best life
  3,214 / 3,952.

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
no food, using a saved checkpoint. Run it on the VM after the corrected
use-inheritance head has been live for enough generations.

## Known issues

- The local SSH tunnel dies during long restarts or network changes. Restart
  the command above. The Cloudflare links do not need the laptop tunnel.
- Cloudflare quick URLs are public and temporary. Do not share them.
- The world model is passive: it predicts the next senses, not the effect of
  each action. The action head is inherited at birth, not learned inside a
  life. There is still no planning.
- Wall spam: the old per-step use growth kept raising forward at a wall even
  when the body did not move. Fixed on 2026-10-31: only effective use is
  recorded (forward that moved, turn, eat that got food). The parent’s head is
  fixed for life; the child gets the boost at birth. Selection must still
  remove wall lineages.
- Pre-body-frame generation files are not valid for the current eval.
- The old best records (88 / 355) are discarded by the epoch reset.
- SOTF children died in every session. Sessions 2-5 each added 24 children;
  all died; the population returned to 8. The earlier sessions ran into the
  lean phase. The plenty phase can fill 32 (8001 hit 32), but the lean phase
  falls to ~8. SOTF stays off until the phase/floor policy is chosen.
- The reset cleared the score boards, not the parents’ action heads. The
  current bodies carry heads grown under the old per-step rule; the new
  effective-use rule applies from birth onward. Selection must remove any
  pre-fix wall lineages.
- The page can look frozen below pop 32: empty seats and a moving SOTF
  counter. The step still advances. SOTF is off now.
- The first SOTF grouping kept empty rooms in the top 16. Fixed: the top and
  near groups are living rooms; empty rooms always refill.

## Decisions

- No reward, no penalty, no controller, no teaching.
- Use-dependent inheritance is allowed: the parent’s effective use is added to
  the child at birth. It uses the chosen action and the active state, not food
  or energy as a reward. The parent’s head is fixed for life.
- The action score is a record only.
- SOTF is off. The 24-child burst failed at pop 8. The plenty phase can reach
  32, but the lean phase falls to ~8; choose the phase/floor policy before
  running another session. The reliable reflex is still the goal.
- The 4 near-miss rooms (ranks 17–20) are the planned experiment group.
- The witness (fixed substrate) comes later.

## Next steps

1. Verify the wall fix: watch whether new children stop spamming forward at
   walls; check the action share over the next generations.
2. Run the eat probe on both instances after enough generations:
   `P(eat | food)` vs `P(eat | none)`. Target above 0.7 vs below 0.2.
3. Decide the SOTF phase policy: the plenty phase reached 32 on 8001, but the
   lean phase falls to ~8. Options: run SOTF at the peak, raise the lean food
   floor, or keep the reflex focus. SOTF cannot work at 8.
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
| 2026-10-31 | SOTF selection raises the eat reflex | food rate of top vs bottom; reflex probe | failed: 5 sessions, 24 children died each round, pop 8; SOTF off | failed at current carrying capacity |
| 2026-10-31 | Use-dependent inheritance at birth removes wall spam | action share; wall escapes; P(eat \| food) | running from reset epochs 920,133 / 943,206 | pending |
| 2026-10-31 | Food-cycle carrying capacity is phase-dependent | population over a cycle | 8001 reached 32 (plenty) while 8000 fell to 8 (lean) | observed; SOTF phase policy pending |
