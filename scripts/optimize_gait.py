import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cma
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import ROOT, load_yaml
from sim.gait import GaitSpec
from sim.parallel import eval_task, init_worker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade", type=int, default=0)
    ap.add_argument("--mode", default="mechanical", choices=["mechanical", "functional"])
    ap.add_argument("--init", default=None)
    ap.add_argument("--free-legs", nargs="*", default=[])
    ap.add_argument("--affected", nargs="*", default=None)
    ap.add_argument("--gens", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sigma0", type=float, default=None)
    ap.add_argument("--speed", type=float, default=None, help="override gait.target_speed (curriculum)")
    ap.add_argument("--calibrate", action="store_true", help="healthy baseline: add control-dog joint window terms")
    ap.add_argument("--load-avoid", type=float, default=0.0, help="H3: shift the allowed hind-limb load share down by this fraction (pain avoidance)")
    ap.add_argument("--ext-override", type=float, default=None, help="sensitivity: set the knee extension limit (stifle included angle, deg) of --grade, both modes")
    ap.add_argument("--seed", type=int, default=None, help="CMA-ES seed (repeat runs)")
    ap.add_argument("--cost-set", nargs="*", default=[], help="cost overrides key=value (floats), e.g. w_shape=5")
    ap.add_argument("--template-mix", type=float, default=1.0, help="blend 0..1 between sinusoid (0) and template (1) for the distal joints")
    ap.add_argument("--template-shift", type=float, nargs=2, default=[0.0, 0.0], help="template time shift (cycles) fore hind")
    ap.add_argument("--template", action="store_true", help="distal joints (carpus/hock) follow the video-derived waveform template")
    ap.add_argument("--gait-set", nargs="*", default=[], help="gait overrides key=value (floats), e.g. eval_duration=8")
    ap.add_argument("--body-set", nargs="*", default=[], help="body composition overrides key=value (muscle_hind, muscle_fore, fat), see sim.dog_scale.apply_composition")
    ap.add_argument("--leo-config", default="leo", help="model config name in configs/model/ used with --leo (e.g. leo_bone)")
    ap.add_argument("--leo", action="store_true", help="Leo-proportioned dog (configs/model/leo.yaml) with rescaled absolute-length thresholds")
    ap.add_argument("--spine", action="store_true", help="enable the 8 spine/scapula parameters (lumbar extension+bend tendons, scapula swing)")
    ap.add_argument("--w-meta", type=float, default=None, help="override cost.w_meta (digitigrade stance penalty)")
    ap.add_argument("--digit-kp", type=float, default=None, help="override servo.digit_kp (stiffness annealing)")
    ap.add_argument("--style-ref", default=None, help="run dir of the healthy gait: H2 keep its joint excursions")
    ap.add_argument("--style-w", type=float, default=2.0)
    a = ap.parse_args()

    cfg = load_yaml("configs/gait/trot.yaml")
    if a.speed is not None:
        cfg["gait"]["target_speed"] = a.speed
    cfg["cost"]["calibrate"] = bool(a.calibrate)
    if a.load_avoid:
        lo, hi = cfg["cost"]["hind_load_range"]
        cfg["cost"]["hind_load_range"] = [max(0.0, lo - a.load_avoid), hi - a.load_avoid]
    if a.leo:
        from sim.dog_scale import apply_leo
        cfg = apply_leo(cfg, a.leo_config)
    if a.spine:
        cfg["gait"]["spine"] = True
    for kv in a.gait_set:
        k, v = kv.split("=")
        if k == "amp_max": cfg["gait"]["amp_bounds"] = [cfg["gait"]["amp_bounds"][0], float(v)]
        else: cfg["gait"][k] = float(v)
    for kv in a.cost_set:
        k, v = kv.split("="); cfg["cost"][k] = float(v)
    for kv in a.body_set:
        k, v = kv.split("="); cfg.setdefault("body", {})[k] = float(v)
    if a.template:
        cfg["gait"]["template"] = True
        cfg["gait"]["template_mix"] = a.template_mix
        cfg["gait"]["template_shift"] = {"fore": a.template_shift[0], "hind": a.template_shift[1]}
    if a.w_meta is not None:
        cfg["cost"]["w_meta"] = a.w_meta
    if a.digit_kp is not None:
        cfg["servo"]["digit_kp"] = a.digit_kp
    if a.style_ref:
        from sim.rollout import Sim
        rc = yaml.safe_load(open(ROOT / "results" / a.style_ref / "trot.yaml", encoding="utf-8"))
        z0 = np.load(ROOT / "results" / a.style_ref / "best.npz", allow_pickle=True)
        cfg_ref = {k_: v_ for k_, v_ in cfg.items() if k_ != "body"}
        s0 = Sim(cfg_ref)
        s0.limiter.set_grade(load_yaml("configs/impairment/patella.yaml"), 0, "mechanical")
        o0 = s0.run(GaitSpec(rc["gait"], list(z0["free_legs"])), z0["x"])
        ref = {j: float(o0["win"][j][1] - o0["win"][j][0]) for j in ("stifle", "hock", "hip")}
        cfg["cost"]["style"] = {"ref": ref, "w": a.style_w, "scale": 10.0}
        print("style reference excursions (deg):", ref, flush=True)
    imp = load_yaml("configs/impairment/patella.yaml")
    oc = cfg["optim"]
    if a.ext_override is not None:
        imp["grades"][a.grade]["prom_ext"] = a.ext_override
        imp["grades"][a.grade]["arom_ext"] = a.ext_override
    affected = a.affected or imp["affected_legs"]
    tag = a.tag or f"opt_g{a.grade}_{a.mode}"
    out = ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    shutil.copy(ROOT / "configs/impairment/patella.yaml", out / "patella.yaml")

    spec = GaitSpec(cfg["gait"], a.free_legs)
    lo, hi = spec.bounds()
    if a.init:
        z = np.load(ROOT / a.init, allow_pickle=True)
        src_cfg = yaml.safe_load(open((ROOT / a.init).parent / "trot.yaml", encoding="utf-8"))["gait"]
        src = GaitSpec(src_cfg, list(z["free_legs"]))
        x0 = spec.warm_start(src, z["x"])
    else:
        x0 = spec.initial_guess()
    u0 = np.clip((x0 - lo) / (hi - lo), 0.01, 0.99)

    rng_seed = a.seed if a.seed is not None else oc["seed"]
    es = cma.CMAEvolutionStrategy(
        u0, a.sigma0 or oc["sigma0"],
        {"bounds": [0, 1], "popsize": oc["popsize"], "seed": rng_seed, "verbose": -9},
    )
    task = lambda u: dict(x=(lo + (hi - lo) * np.asarray(u)).tolist(), free_legs=a.free_legs,
                          grade=a.grade, mode=a.mode, affected=affected)
    gens = a.gens or oc["max_gens"]
    hist, best = [], (np.inf, None, None)
    t0 = time.time()
    with ProcessPoolExecutor(oc["workers"], initializer=init_worker, initargs=(cfg, imp)) as ex:
        if a.init:
            c0, s0 = eval_task(dict(x=x0.tolist(), free_legs=a.free_legs, grade=a.grade, mode=a.mode, affected=affected))                 if False else ex.submit(eval_task, dict(x=x0.tolist(), free_legs=a.free_legs, grade=a.grade,
                                                        mode=a.mode, affected=affected)).result()
            best = (c0, np.asarray(x0), s0)
            print(f"start point cost {c0:.3f}", flush=True)
        for g in range(gens):
            U = es.ask()
            res = list(ex.map(eval_task, [task(u) for u in U]))
            costs = [r[0] for r in res]
            es.tell(U, costs)
            i = int(np.argmin(costs))
            if costs[i] < best[0]:
                best = (costs[i], lo + (hi - lo) * np.asarray(U[i]), res[i][1])
            hist.append(dict(gen=g, best_cost=min(costs), median_cost=float(np.median(costs)),
                             frac_fell=float(np.mean([r[1]["fell"] for r in res])),
                             v_best=res[i][1]["v_mean"], sigma=es.sigma))
            if g % 5 == 0 or g == gens - 1:
                h = hist[-1]
                print(f"gen {g:3d} best={h['best_cost']:.3f} med={h['median_cost']:.2f} fell={h['frac_fell']:.2f} "
                      f"v={h['v_best']:.2f} sigma={h['sigma']:.3f} ({time.time() - t0:.0f}s)", flush=True)
                pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
                np.savez(out / "best.npz", x=best[1], cost=best[0], free_legs=np.array(a.free_legs, dtype=object),
                         grade=a.grade, mode=a.mode, affected=np.array(affected, dtype=object), summary=str(best[2]))
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    print("BEST", best[0], best[2])
    print("saved", out / "best.npz")
    (out / "done.txt").write_text("ok", encoding="utf-8")


if __name__ == "__main__":
    main()
