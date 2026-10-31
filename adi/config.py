from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Everything that is fixed before the first birth.

    The body plan and the world rules live here. No wants live here.
    """

    # world
    grid: int = 48
    fertility: float = 0.05
    food_regen: float = 0.0010
    max_range: float = 6.0
    ray_steps: int = 12

    # population
    n_slots: int = 32
    init_pop: int = 20
    min_pop: int = 12

    # body
    obs_dim: int = 14
    act_dim: int = 4
    move_speed: float = 0.40
    turn_rate: float = 0.38
    metabolic: float = 0.06
    bite: float = 28.0
    start_energy: float = 70.0
    child_energy: float = 55.0
    birth_energy: float = 90.0

    # brain
    hidden: int = 128
    lr: float = 3e-4
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    mut_p_body: float = 0.002
    mut_s_body: float = 0.02
    mut_p_pol: float = 0.02
    mut_s_pol: float = 0.05

    # runtime
    seed: int = 7
    base_hz: float = 30.0
    snap_hz: float = 30.0
