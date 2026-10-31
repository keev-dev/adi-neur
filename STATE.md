# STATE — handoff

Last updated: 2026-10-31. This file is the restart point. Read it first.

## What this is

A small embodied agent (“Adi”) on an Oracle VM. There is no reward, no
teaching, and no hand-written controller. Bodies live in rooms, eat, die, and
are copied. Selection does the work. The current goal is a **reliable eat
reflex**. SOTF is on 8000, timed to the plenty phase; 8001 is the continuous
control. On 2026-10-31 the action head moved to use-dependent inheritance at
birth, and the score boards were reset so the next records belong to this
mechanism. The post-reset population swings from 8 in the lean phase to 32 in
the plenty phase. Session 6 (step 967,929) refilled the 24 empty rooms at the
plenty start, but all 24 children died in the fall/lean. Session 7 (step
973,929) filled 32 again; its 24 children face the next lean. The 8001 control
peaked at 32 and is at 22 in the lean. The UI has a live SOTF panel.

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

The viewer has a SOTF strip (next session, last top 3). `/stats` has the full
SOTF section: next session, last-session bests and rows, group counts, and the
live contenders.

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
- SOTF: **on 8000** since 2026-10-31, interval 6,000. Session 6 ran at step
  967,929 (plenty start) filled 32, but all 24 children died in the fall/lean.
  Session 7 (step 973,929) filled 32 again; its 24 children face the next
  lean. Session 8 is due at step 979,929. Port 8001 is the continuous control
  (SOTF off); it peaked at 32 and is at 22 in the lean.
- Birth rule: `threshold` (replace weakest below 15 energy only).
- Records reset epoch: 920,133 on port 8000; 943,206 on port 8001 (reset
  2026-10-31 after the use-inheritance deploy). Only bodies born after these
  steps count for `best eat` / `best life` and the per-seat means. The old
  88 / 355 records are intentionally discarded. Current new records:
  51 / 11,580 on 8000; 90 / 21,841 on 8001.
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
- Session 6, step 967,929: plenty start. 8 kept; 24 empty rooms refilled from
  the top 8; population 32. All 24 children died in the fall/lean; 8000 back
  to 8. Top: Yama 7.62, Agni 4.91, Indra 4.60. Session 7 followed at step
  973,929.
- Session 7, step 973,929: plenty start. 8 kept; 24 empty rooms refilled from
  the top 8; population 32. Top: Brahma 5.53, Surya 5.00, Varuna 4.48. The 24
  children face the next lean. Session 8 is due at step 979,929.
- SOTF is on 8000; 8001 is the control. The use-inheritance code is live on
  both. The population swings with the food cycle: 8000 filled 32 at the
  plenty start, fell to 8 in the lean, then session 7 filled 32 again. The
  8001 control peaked at 32 and was at 22 in the lean at step ~998k. New
  records: 8000 51 eaten / 11,580 life; 8001 90 / 21,841.

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
# schedule the first session after `delay` steps:
curl -X POST http://127.0.0.1:8000/cmd -H 'Content-Type: application/json' \
  -d '{"cmd":"sotf","value":1,"interval":6000,"delay":1069}'
# SOTF panel data:
curl -s http://127.0.0.1:8000/sotf
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
- SOTF children died in every session. Sessions 2-6 filled rooms, but all 24
  new children died by the lean. Session 7 (step 973,929) filled 32 again; its
  24 children face the next lean. The 8001 control peaked at 32 and is at 22
  in the lean, best 90 / 21,841. The SOTF children are not surviving; session
  8 is the next test.
- The reset cleared the score boards, not the parents’ action heads. The
  current bodies carry heads grown under the old per-step rule; the new
  effective-use rule applies from birth onward. Selection must remove any
  pre-fix wall lineages.
- The page can look frozen below pop 32: empty seats and a moving SOTF
  counter. The step still advances. SOTF is on 8000; 8001 is off.
- The first SOTF grouping kept empty rooms in the top 16. Fixed: the top and
  near groups are living rooms; empty rooms always refill.

## Decisions

- No reward, no penalty, no controller, no teaching.
- Use-dependent inheritance is allowed: the parent’s effective use is added to
  the child at birth. It uses the chosen action and the active state, not food
  or energy as a reward. The parent’s head is fixed for life.
- The action score is a record only.
- SOTF is on 8000, timed to the plenty start, interval 6,000; 8001 is the
  control. Sessions 6 and 7 filled 32 seats, but the session 6 children all
  died in the fall/lean. Session 7's children are the next test; if they fail,
  turn SOTF off or fix the child/lean energy. The reliable reflex is still the
  goal.
- SOTF stats are part of the UI: the viewer strip shows the next session and
  the last top 3; the stats page shows the last session, the bests, the group
  counts, and the live contenders.
- The 4 near-miss rooms (ranks 17–20) are the planned experiment group.
- The witness (fixed substrate) comes later.

## Next steps

1. Session 6's 24 children all died in the fall/lean. Session 7 (step
   973,929) filled 32 again; its 24 children face the next lean. Watch
   session 8 at step 979,929. If the children fail, turn SOTF off or fix the
   child/lean energy. The 8001 control is at 22 in the lean. The SOTF panel
   shows the top performers and group transitions.
2. Verify the wall fix: watch whether new children stop spamming forward at
   walls; check the action share over the next generations.
3. Run the eat probe on both instances after enough generations:
   `P(eat | food)` vs `P(eat | none)`. Target above 0.7 vs below 0.2.
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
| 2026-10-31 | Food-cycle carrying capacity is phase-dependent | population over a cycle | 8001 control peaked at 32, 22 in the lean; 8001 90 / 21,841, 8000 51 / 11,580 | observed; SOTF filled 32, children died by the lean |
| 2026-10-31 | SOTF at the plenty start can fill 32 seats | session 6 population | 24 children, pop 32; top Yama 7.62; all 24 died by the lean; 8000 back to 8 | filled 32, failed lean survival |
| 2026-10-31 | SOTF session 7 fills 32 at the plenty start | session 7 population | 24 children, pop 32; top Brahma 5.53 | observed; lean survival pending |
| 2026-10-31 | SOTF stats in the UI show the best performers | viewer strip + stats page | live; last session and live contenders | done |
