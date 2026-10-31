"""Files on disk. The trace is the evidence.

genome.csv   one row for each birth
events.csv   one row for each birth and death
traits.csv   header only. rows are added only after probing.
trace.jsonl  the focused line, one line every N steps
"""

import atexit
import csv
import json
from pathlib import Path


class Logger:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        events = self.root / "events.csv"
        genome = self.root / "genome.csv"
        seats = self.root / "seat_events.csv"
        traits = self.root / "traits.csv"
        self.trace_path = self.root / "trace.jsonl"

        new_events = not events.exists()
        new_genome = not genome.exists()
        new_seats = not seats.exists()

        self.f_events = open(events, "a", newline="", encoding="utf-8")
        self.f_genome = open(genome, "a", newline="", encoding="utf-8")
        self.f_seats = open(seats, "a", newline="", encoding="utf-8")
        self.f_trace = open(self.trace_path, "a", encoding="utf-8")

        if new_events:
            csv.writer(self.f_events).writerow(
                ["step", "event", "id", "parent", "gen", "age", "energy", "eaten"])
        if new_genome:
            csv.writer(self.f_genome).writerow(
                ["step", "id", "parent", "gen", "energy"])
        if new_seats:
            csv.writer(self.f_seats).writerow(
                ["step", "event", "seat", "id", "parent", "gen", "age", "energy", "eaten"])
        if not traits.exists():
            with open(traits, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["id", "name", "probe", "value", "note", "step"])

        atexit.register(self.close)

    def genome(self, step, aid, parent, gen, energy):
        csv.writer(self.f_genome).writerow([step, aid, parent, gen, round(energy, 2)])

    def event(self, step, kind, aid, parent, gen, age, energy, eaten):
        csv.writer(self.f_events).writerow(
            [step, kind, aid, parent, gen, age, round(energy, 2), int(eaten)])

    def seat_event(self, step, kind, seat, aid, parent, gen, age, energy, eaten):
        csv.writer(self.f_seats).writerow(
            [step, kind, seat, aid, parent, gen, age, round(energy, 2), int(eaten)])

    def trace(self, obj: dict):
        self.f_trace.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def flush(self):
        for f in (self.f_events, self.f_genome, self.f_seats, self.f_trace):
            f.flush()

    def close(self):
        for f in (self.f_events, self.f_genome, self.f_seats, self.f_trace):
            try:
                f.close()
            except Exception:
                pass
