import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.metrics import summarize
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec
from sim.rollout import Sim


def evaluate(run: str, duration: float | None = None, quiet: bool = False) -> dict:
    d = ROOT / "results" / run
    cfg = yaml.safe_load(open(d / "trot.yaml", encoding="utf-8"))
    pcopy = d / "patella.yaml"
    imp = yaml.safe_load(open(pcopy, encoding="utf-8")) if pcopy.exists() and "sens" in run else load_yaml("configs/impairment/patella.yaml")
    z = np.load(d / "best.npz", allow_pickle=True)
    spec = GaitSpec(cfg["gait"], list(z["free_legs"]))
    sim = Sim(cfg)
    sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
    o = sim.run(spec, z["x"], record=True, duration=duration)
    s = summarize(o)
    s.update(run=run, grade=int(z["grade"]), mode=str(z["mode"]), target_speed=cfg["gait"]["target_speed"],
             affected=list(z["affected"]), cost=o["cost"])
    with open(d / "summary.json", "w") as f:
        json.dump(s, f, indent=1, default=float)
    np.savez_compressed(d / "log.npz", **{k: v for k, v in o.items() if isinstance(v, np.ndarray)},
                        freq=o["freq"], warmup=o["warmup"])
    if not quiet:
        keys = ["fell", "speed", "freq_hz", "stride_len", "LH_stifle_rom", "LH_stifle_max", "LH_stifle_min",
                "LH_hip_rom", "LH_hock_rom", "LH_duty", "LF_duty"]
        print({k: (round(s[k], 2) if isinstance(s[k], float) else s[k]) for k in keys if k in s})
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--duration", type=float, default=None)
    a = ap.parse_args()
    evaluate(a.run, a.duration)
