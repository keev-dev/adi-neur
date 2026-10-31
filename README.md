# Adi

A small embodied line. No reward function. No language model. No human data.

Adi is a population of tiny recurrent brains in a room. Energy falls. Food
raises it. At zero the body stops. Above a threshold the body splits and the
child inherits the weights, with noise. There is no reward for food. There is
no "eat" objective. The urge to eat is what remains of the lines that did not
stop.

## What runs

- One private 48x48 room per body. Walls, food, partial sight.
- A body with 14 senses: 8 whiskers, local food, food gradient in the body frame
  (ahead, left), heading, energy.
- Four actions: forward, left, right, eat.
- A 128-unit GRU. The policy head is frozen inside a life. The recurrent body
  and the prediction head learn, every step, to predict the next observation.
  That is the only learning signal. It is self-supervised and has no reward.
- A population of up to 32. Energy falls. Food raises it. Zero energy stops the
  body. Above 90, the body splits. If the room is full, the child replaces the
  weakest living body. The child gets a mutated copy of the parent's weights.

## Run

```
pip install -r requirements.txt
python run.py
```

Open http://127.0.0.1:8000

The simulation starts at 1x. Use the buttons to run faster. `max` is about
100 steps per second on this machine.

More than one instance:

```
python run.py --instance 1     # port 8000, seed 7
python run.py --instance 2     # port 8001, seed 8
python launch.py 2             # two instances under one supervisor
python run.py --fresh          # ignore the checkpoint and start new
```

Each instance is independent and uses one core. Do not thread one instance.
For deployment, see `deploy/README.md`.

## The two panes

Left: the room as it is.

Right: the room as one line represents it.

The gap between the panes is the experiment.

- **Recurrent state** — the 128 hidden units, live. Amber is positive, blue is
  negative.
- **Policy** — the four action probabilities, plus the prediction error over
  the last 240 steps.
- **Attention** — the mean absolute gradient of the chosen action with respect
  to each input group. It shows what changed the action just now.
- **State map** — a fixed random projection of the hidden state, one point per
  step. Colour is the situation: grey normal, green food, blue wall, red low
  energy. Clusters form as the world model learns. The map resets when the
  focused line dies.

## Names, colours, and the roster

The 32 seats have names. A name belongs to the seat, not the body. Bodies die.
The seat remains.

Each seat has a hue. Its brightness mixes three things: current energy, recent
foraging, and lineage depth. A body flashes when it eats.

The roster under the room shows all 32 seats. Click a seat to focus it. The
`auto` button returns to the newest line.

The birth-rule select turns competitive birth on or off and chooses the
replacement rule. With it off, births happen only when a seat is free. The
population may then freeze at the cap. Compare the regimes.

## Seat table and per-seat stats

- **/kali** — a table of the living bodies. Seat, name, id, generation, age,
  energy, cells eaten, rate, coverage, path, action mix, guess error. Sort by
  food, coverage, rate, or generation. Click a row to focus that seat on the map.
- **/stats** — one row per seat: births, deaths, mean life, max life, mean
  generation, mean cells eaten per body, best cells eaten, and the current body.

Coverage is the fraction of the room the body has visited. Path is the distance
it has moved. Both start at zero when a body is born.

The birth rule has four modes, selected in the header:

- `weakest` — a rich body replaces the weakest living body.
- `threshold` — a rich body replaces the weakest only if its energy is below 15.
- `oldest` — a rich body replaces the oldest living body.
- `off` — no replacement.

`threshold` is the default for new runs. It is the slower, less churny rule.

## Files

| Path | Holds |
|---|---|
| `data/runs/<run_id>/genome.csv` | one row per birth: id, parent, generation, energy |
| `data/runs/<run_id>/events.csv` | births and deaths |
| `data/runs/<run_id>/traits.csv` | header only. Add a row only after probing. |
| `data/runs/<run_id>/trace.jsonl` | the focused line, every 20 steps |
| `data/runs/<run_id>/hall_of_fame/` | weights of the best foragers by cells eaten |
| `data/runs/<run_id>/run_meta.json` | the exact config of the run |

`traits.csv` stays empty until a trait is found in the trace. Name after, not
before. Naming before is projection.

## Measured now

- prediction error (self-supervised)
- energy, age, cells eaten
- generation depth
- population, births, deaths, mean lifetime

## Not built yet

- the witness: fixed parameters, dynamic state, shared, never trained by reward,
  never able to act
- the precons probe set
- more than one body in the same room
- teaching and communication

## Checkpoints

Every 10 minutes, and on SIGTERM, the simulation writes a checkpoint:

```
data/checkpoints/instance_<n>.pt
```

It holds the population, the weights, the Adam state, the rooms, the RNG, the
focus, and the step. On startup, `run.py` loads it and continues. Use `--fresh`
to ignore it.

The deploy script uses this:

```
bash deploy/pull.sh --restart
```

It pauses, checkpoints, pulls, restarts, and resumes.

## Rule

Hardcode the world and the body. Do not hardcode the wants.

## Notes

- v0 runs on CPU. One torch thread is faster than many for tensors this small.
- On Windows, the timer resolution is set to 1 ms so the speed control is real.
- Torch and numpy are seeded. A run is reproducible from the seed. Mutation and
  action sampling use the torch RNG.

## Ledger

Write one row before each run. Write what counts as success and what counts as
failure. If the result is unclear, write unclear.

| Date | Hypothesis | Metric | Result | Verdict |
|---|---|---|---|---|
| | | | | |
