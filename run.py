import argparse
import ctypes
import dataclasses
from pathlib import Path

import torch
import uvicorn

from adi.config import Config
from adi.server import create_app
from adi.sim import Simulation


def parse_args():
    p = argparse.ArgumentParser(description="Adi simulation")
    p.add_argument("--instance", type=int, default=None,
                   help="instance number. 1 -> port 8000 seed 7, 2 -> port 8001 seed 8")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--root", default=None, help="data root. default: project folder")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--speed", type=float, default=1.0,
                   help="start speed. 0 means as fast as possible")
    p.add_argument("--checkpoint", default=None, help="checkpoint file path")
    p.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    return p.parse_args()


def main():
    args = parse_args()

    if args.instance is not None:
        port = args.port or (8000 + args.instance - 1)
        seed = args.seed if args.seed is not None else (7 + args.instance - 1)
    else:
        port = args.port or 8000
        seed = args.seed if args.seed is not None else 7

    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

    torch.set_num_threads(max(1, args.threads))
    cfg = dataclasses.replace(Config(), seed=seed)
    root = Path(args.root) if args.root else Path(__file__).resolve().parent
    ckpt = (Path(args.checkpoint) if args.checkpoint
            else root / "data" / "checkpoints" / f"instance_{args.instance or 1}.pt")

    sim = Simulation(cfg, root=root, checkpoint_path=ckpt, fresh=args.fresh)
    sim.speed = args.speed
    sim.start()

    app = create_app(sim)
    print(f"adi instance={args.instance} seed={seed} port={port} root={sim.run_dir} resumed={sim.resumed}")
    uvicorn.run(app, host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
