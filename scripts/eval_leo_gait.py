import sys
sys.path.insert(0, ".")
import numpy as np, yaml
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec
from sim.rollout import Sim
imp = load_yaml("configs/impairment/patella.yaml")
CTRL = dict(stifle=(121.8, 167.1), hock=(105.8, 157.4), hip_rom=19.0)
for run in sys.argv[1:]:
    d0 = ROOT / "results" / run; cfg = yaml.safe_load(open(d0 / "trot.yaml", encoding="utf-8")); z = np.load(d0 / "best.npz", allow_pickle=True)
    cfg["cost"]["w_ref"] = 1.0 if cfg["cost"].get("ref_file") else 0
    sim = Sim(cfg); sim.limiter.set_grade(imp, int(z["grade"]), str(z["mode"]), tuple(z["affected"]))
    for dur in (20.0,):
        o = sim.run(GaitSpec(cfg["gait"], list(z["free_legs"])), z["x"], duration=dur); w = o["win"] or {}
        if o["fell"]: print(f"{run}: FELL at {o['t_fall']:.1f} s"); continue
        st, hk = w["stifle"], w["hock"]
        print(f"== {run} ({dur:.0f} s, freq {float(z['x'][0]):.2f} Hz)")
        print("   speed %.2f m/s | yaw %.1f deg/s | td_lag %+.3f cycles (fore - hind; <0 fore first) | duty %s" % (o["v_mean"], o["yaw_rate"], o["td_lag"], np.round(o["duty"], 2)))
        print("   stifle (%.0f, %.0f) vs control (%.0f, %.0f) | hock (%.0f, %.0f) vs (%.0f, %.0f) | hip joint ROM %.0f (control %.0f)" % (st[0], st[1], *CTRL["stifle"], hk[0], hk[1], *CTRL["hock"], o["hip_joint_rom"], CTRL["hip_rom"]))
        print("   hind/fore swing clearance cm (LF RF LH RH) %s | lumbar mean %+.0f deg, torso-pelvis pitch %+.0f deg | paw outside joint cm %s | stride %.2f s (Leo trot 0.41 s at 0.85 m/s)" % (np.round(100 * np.array(o["clearance"]), 1), o["lumb_mean"], o["pdiff_mean"], np.round(100 * np.array(o["track_excess"]), 1), 1.0 / float(z["x"][0])))
        print("   metatarsus stance %.0f deg | digit pitch %s | fore ROM sh/el/wr %s | carpus ROM %s ref_err %s" % (o["meta_mean"], np.round(o["paw_pitch"], 0) if o.get("paw_pitch") else None, np.round(o["fore_rom"], 0), round(o.get("carpus_rom", 0), 0), round(o["ref_err"], 2) if o.get("ref_err") is not None else None))
        print("   hind swing digit pitch p5 (<-45 = sole visible from behind) %s" % (np.round(o["swing_pitch"], 0) if o.get("swing_pitch") else None))
