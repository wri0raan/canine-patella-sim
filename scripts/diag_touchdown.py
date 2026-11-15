import sys
from pathlib import Path
import mujoco, numpy as np, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec, TrotController
from sim.rollout import Sim
imp = load_yaml("configs/impairment/patella.yaml")
for run in sys.argv[1:]:
    d0 = ROOT / "results" / run
    cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
    sim = Sim(cfg); sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
    o = sim.run(GaitSpec(cfg["gait"], list(z["free_legs"])), z["x"], record=True)
    t, grf, f0, t0 = o["time"], o["grf"], o["freq"], 2.0
    legs = ("LF", "RF", "LH", "RH"); td = {}
    for k, leg in enumerate(legs):
        st = grf[:, k] > 2.0
        ev = np.nonzero(~st[:-1] & st[1:])[0]; ev = ev[t[ev] > t0 + 1.0]
        td[leg] = (t[ev] * f0) % 1.0
    def lag(a, b):
        d = [((pa - pb + 0.5) % 1.0) - 0.5 for pa in td[a] for pb in td[b]]
        return float(np.median([x for x in d if abs(x) < 0.25])) if d else float("nan")
    print(f"== {run}: touchdown phase LF {np.median(td['LF']):.2f} RF {np.median(td['RF']):.2f} LH {np.median(td['LH']):.2f} RH {np.median(td['RH']):.2f}")
    print(f"   diagonal LF vs RH: fore minus hind = {lag('LF','RH'):+.3f} cycles ; RF vs LH: {lag('RF','LH'):+.3f}  (negative = fore lands first)")
