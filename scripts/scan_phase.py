import argparse, sys, shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec
from sim.parallel import eval_task, init_worker
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("init"); ap.add_argument("tag")
    ap.add_argument("--w-toeoff", type=float, default=3.0); ap.add_argument("--w-meta", type=float, default=10.0)
    ap.add_argument("--span", type=float, default=0.3); ap.add_argument("--step", type=float, default=0.025)
    ap.add_argument("--speed", type=float, default=0.9)
    ap.add_argument("--fore", action="store_true", help="scan the time shift of the whole FORE group (tau_knee column) instead")
    ap.add_argument("--cost-set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_yaml("configs/gait/trot.yaml"); cfg["gait"]["target_speed"] = a.speed
    for kv in a.cost_set:
        k_, v_ = kv.split("="); cfg["cost"][k_] = float(v_)
    cfg["cost"]["calibrate"] = True; cfg["cost"]["w_meta"] = a.w_meta; cfg["cost"]["w_toeoff"] = a.w_toeoff
    imp = load_yaml("configs/impairment/patella.yaml")
    z = np.load(ROOT / "results" / a.init / "best.npz", allow_pickle=True)
    spec = GaitSpec(cfg["gait"], []); x0 = np.asarray(z["x"], float)
    s = 1 + spec.pg * 1
    shifts = np.round(np.arange(-a.span, a.span + 1e-9, a.step), 4)
    def shifted(tk, th):
        x = x0.copy()
        if a.fore:
            sf = 1
            for joint in range(3):
                x[sf + 3 + joint] += 2 * np.pi * tk; x[sf + 13 + joint] += 4 * np.pi * tk
            wrap = lambda v: (v + np.pi) % (2 * np.pi) - np.pi
            for i in range(sf + 3, sf + 6): x[i] = wrap(x[i])
            for i in range(sf + 13, sf + 16): x[i] = wrap(x[i])
            return x
        for joint, tau in ((1, tk), (2, th)):
            x[s + 3 + joint] += 2 * np.pi * tau; x[s + 13 + joint] += 4 * np.pi * tau
        wrap = lambda v: (v + np.pi) % (2 * np.pi) - np.pi
        for i in (s + 4, s + 5, s + 14, s + 15): x[i] = wrap(x[i])
        return x
    cases = [(tk, 0.0) for tk in shifts] if a.fore else [(tk, th) for tk in shifts for th in shifts]
    tasks = [dict(x=shifted(tk, th).tolist(), free_legs=[], grade=0, mode="functional", affected=["LH", "RH"]) for tk, th in cases]
    with ProcessPoolExecutor(12, initializer=init_worker, initargs=(cfg, imp)) as ex:
        res = list(ex.map(eval_task, tasks, chunksize=4))
    rows = []
    for (tk, th), (c, sm) in zip(cases, res):
        w = sm["win"] or {}
        rows.append(dict(tau_knee=tk, tau_hock=th, cost=c, fell=sm["fell"], v=sm["v_mean"], toeoff=sm.get("toeoff_pen"), tdlag=sm.get("td_lag"), retract=sm.get("retract_pen"), peak=sm.get("peak_pen"), meta=sm["meta_mean"],
                         stifle_lo=w.get("stifle", (np.nan,) * 2)[0], stifle_hi=w.get("stifle", (np.nan,) * 2)[1],
                         hock_lo=w.get("hock", (np.nan,) * 2)[0], hock_hi=w.get("hock", (np.nan,) * 2)[1]))
    R = pd.DataFrame(rows).sort_values("cost"); out = ROOT / "results" / a.tag; out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "scan.csv", index=False)
    print(R.head(8).round(2).to_string()); print("start (0,0):", R[(R.tau_knee == 0) & (R.tau_hock == 0)].round(2).to_string(header=False))
    b = R.iloc[0]; xb = shifted(b.tau_knee, b.tau_hock)
    np.savez(out / "best.npz", x=xb, cost=b.cost, free_legs=np.array([], dtype=object), grade=0, mode="functional", affected=np.array(["LH", "RH"], dtype=object), summary="scan")
    yaml.safe_dump(cfg, open(out / "trot.yaml", "w", encoding="utf-8"), sort_keys=False); shutil.copy(ROOT / "configs/impairment/patella.yaml", out / "patella.yaml")


if __name__ == "__main__":
    main()
