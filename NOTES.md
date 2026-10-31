# operational notes

## run
```
python run.py
```
http://127.0.0.1:8000

## commands (HTTP)
```
POST /cmd  {"cmd":"speed","value":0}   # 0 = max
POST /cmd  {"cmd":"pause"}             # toggle
```

## smoke test
```
python -c "from adi.config import Config; from adi.sim import Simulation; s=Simulation(Config()); [s.step_once() for _ in range(500)]; print(s.pop.stats())"
```

## what to probe first

1. Does prediction error fall inside one life?
2. Does the policy distribution move away from uniform across generations?
3. Do high-generation lines eat more than low-generation lines?
4. Does the state map separate food from wall states?

## what would falsify the current design

- prediction error does not fall
- policy stays at 0.25 across all actions for 50+ generations
- state map never separates situations
- population only survives because of the min_pop refill

If the last one holds, the ecology is not viable and the tuning is wrong,
not the idea.
