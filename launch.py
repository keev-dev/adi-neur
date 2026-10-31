"""Start N independent instances, one per core. Restart a child if it exits."""

import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
procs: list[subprocess.Popen] = []
running = True


def shutdown(*_):
    global running
    running = False
    for p in procs:
        if p.poll() is None:
            p.terminate()


def spawn(i: int) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(HERE / "run.py"), "--instance", str(i)],
        cwd=str(HERE),
    )


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    for i in range(1, n + 1):
        procs.append(spawn(i))
    print(f"launched {n} instances on ports {8000}..{8000 + n - 1}")

    while running:
        for idx, p in enumerate(list(procs)):
            if p.poll() is not None and running:
                print(f"child {p.pid} exited with {p.returncode}; restarting instance {idx + 1}")
                procs[idx] = spawn(idx + 1)
        time.sleep(2)

    for p in procs:
        if p.poll() is None:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
