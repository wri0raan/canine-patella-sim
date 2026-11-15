import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.ensemble import RandomForestClassifier
from analysis.video_style_features import video_landmarks, features, sim_landmarks, windows
from analysis.identify_angles import (video_L, angle_series, feats, FEATS, sim_L, windows_of)
from analysis.identify import ALL

SEG = {"n1": (0.0, 0.9), "n2": (1.1, 2.5), "n4": (0.4, 1.7), "n5": (0.6, 2.1)}
D = pd.read_csv("results/ident_results/dataset.csv"); gr = D[(D.hyp != "body") | D.cond.str.match(r"idg_M(10|08)F00_S")].copy()
gr["knee"] = np.select([gr.grade == 0, gr.grade == 2], ["normal", "abnormal"], "severe")
mk = lambda d, f: RandomForestClassifier(300, min_samples_leaf=2, random_state=0, class_weight="balanced").fit(d[f], d.knee)
SF = ["speed", "stride_hz", "step"]
mS = mk(gr, SF)
print("training range (base grade runs): speed %.2f-%.2f m/s, stride %.2f-%.2f Hz, step %.2f-%.2f" % (gr.speed.min(), gr.speed.max(), gr.stride_hz.min(), gr.stride_hz.max(), gr.step.min(), gr.step.max()))
rows = []
lab = dict(zip(gr.run, gr.grade))
for r in sorted(gr.run.unique()):
    try: L, fps = sim_L(f"results/ident/{r}.npz")
    except Exception: continue
    for w in windows_of(angle_series(L, fps), fps): rows.append(dict(knee="normal" if lab[r] == 0 else "abnormal" if lab[r] == 2 else "severe", **w))
for ch in "AB":
    for suf, kn in (("", "normal"), ("_G2", "abnormal"), ("_G3", "severe")):
        try: L, fps = sim_L(f"results/ident/fit{ch}{suf}.npz")
        except Exception: continue
        for w in windows_of(angle_series(L, fps), fps): rows.append(dict(knee=kn, **w))
FA = pd.DataFrame(rows)
mA1 = mk(FA, ["stif_mean"]); mA4 = mk(FA, FEATS)
out = []
for tag, (t0, t1) in SEG.items():
    z = np.load(f"data/real/leo3/work/{tag}_kp.npz"); kp, names = z["kp"], list(z["names"]); a, b = int(t0 * 120), int(t1 * 120) - 1
    f = features(video_landmarks(kp, names, a, b), 120)
    A = angle_series(video_L(kp, names, a, b), 120); af = feats(A, 0, len(A["hock"]))
    ps = dict(zip(mS.classes_, mS.predict_proba(pd.DataFrame([{c: f[c] for c in SF}]))[0]))
    p1 = dict(zip(mA1.classes_, mA1.predict_proba(pd.DataFrame([{"stif_mean": af["stif_mean"]}]))[0]))
    p4 = dict(zip(mA4.classes_, mA4.predict_proba(pd.DataFrame([{c: af[c] for c in FEATS}]))[0]))
    inrange = bool(gr.speed.min() <= f["speed"] <= gr.speed.max() and gr.stride_hz.min() <= f["stride_hz"] <= gr.stride_hz.max())
    out.append(dict(clip=tag, seconds=t1 - t0, speed=f["speed"], stride_hz=f["stride_hz"], step=f["step"], in_training_range=inrange, hock_mean=af["hock_mean"], stif_mean=af["stif_mean"], stif_rom=af["stif_rom"],
                    S_normal=ps.get("normal", 0), S_abn=ps.get("abnormal", 0), S_sev=ps.get("severe", 0), S_call=max(ps, key=ps.get),
                    A1_normal=p1.get("normal", 0), A1_abn=p1.get("abnormal", 0), A1_sev=p1.get("severe", 0), A1_call=max(p1, key=p1.get),
                    A4_normal=p4.get("normal", 0), A4_abn=p4.get("abnormal", 0), A4_sev=p4.get("severe", 0), A4_call=max(p4, key=p4.get)))
R = pd.DataFrame(out); R.to_csv("results/ident_results/M_leo3.csv", index=False)
pd.set_option("display.width", 250); print(R.round(2).T.to_string())
