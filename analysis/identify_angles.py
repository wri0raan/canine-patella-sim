import sys
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
sys.path.insert(0, ".")
from sklearn.ensemble import RandomForestClassifier
from analysis.video_style_features import angle2, video_landmarks

FC = 6.0; WIN = 3.0; HOP = 1.0; SKIP = 2.0
JOINTS = ["hock", "stif", "carp", "elb"]
FEATS = [f"{j}_{s}" for j in JOINTS for s in ("mean", "rom")]
SETS = {"stifle mean only": ["stif_mean"], "means only (4 joints)": [f"{j}_mean" for j in JOINTS], "stifle": ["stif_mean", "stif_rom"], "stifle+hock": ["stif_mean", "stif_rom", "hock_mean", "hock_rom"], "all angles": FEATS,
        "fore only (control)": ["carp_mean", "carp_rom", "elb_mean", "elb_rom"]}


def lp(x, fps):
    b, a = butter(3, FC / (fps / 2)); return filtfilt(b, a, x, axis=0)


def sim_L(path):
    z = np.load(path, allow_pickle=True); n = list(z["names"][: len(z["names"]) - 4]); P = z["pos"]; fps = float(z["fps"]); ix = {k: i for i, k in enumerate(n)}
    xz = lambda k: P[:, ix[k]][:, [0, 2]]; tip = lambda k: P[:, P.shape[1] - 4 + k][:, [0, 2]]
    return dict(withers=xz("C_7"), rump=xz("Ca_1"), stifle=xz("lower_leg_L"), hock=xz("foot_L"), hpaw=xz("foot_anchor_L"), elbow=xz("lower_arm_L"), carpus=xz("hand_anchor_L"), fpaw=tip(0)), fps


def video_L(kp, names, a, b):
    V = video_landmarks(kp, names, a, b)
    return dict(withers=V["back_a"], rump=V["back_b"], stifle=V["stifle"], hock=V["hock"], hpaw=V["hpaw"], elbow=V["elbow"], carpus=V["carpus"], fpaw=V["fpaw"])


def angle_series(L, fps):
    L = {k: lp(v, fps) for k, v in L.items()}
    return dict(hock=angle2(L["stifle"], L["hock"], L["hpaw"]), stif=angle2(L["rump"], L["stifle"], L["hock"]),
                carp=angle2(L["elbow"], L["carpus"], L["fpaw"]), elb=angle2(L["withers"], L["elbow"], L["carpus"]))


def feats(A, a, b):
    o = {}
    for j in JOINTS:
        v = A[j][a:b]; o[j + "_mean"] = float(v.mean()); o[j + "_rom"] = float(np.percentile(v, 97) - np.percentile(v, 3))
    return o


def windows_of(A, fps, win=WIN, hop=HOP, skip=SKIP):
    n = len(A["hock"]); w = int(win * fps); h = int(hop * fps); return [feats(A, a, a + w) for a in range(int(skip * fps), n - w + 1, h)]


NOISE_DEG = 8.0


def noisy(X, f, scale, rng):
    X = X[f].copy()
    for k in f: X[k] = X[k] + rng.normal(0, NOISE_DEG * scale, len(X))
    return X


if __name__ == "__main__":
    D = pd.read_csv("results/ident_results/dataset.csv"); gr = D[(D.hyp != "body") | D.cond.str.match(r"idg_M(10|08)F00_S")]
    runs = sorted(gr.run.unique()); lab = dict(zip(gr.run, gr.grade)); rows = []
    for r in runs:
        try: L, fps = sim_L(f"results/ident/{r}.npz")
        except Exception as e: print("skip", r, e); continue
        for w in windows_of(angle_series(L, fps), fps): rows.append(dict(run=r, chain="base", grade=lab[r], knee="normal" if lab[r] == 0 else "abnormal" if lab[r] == 2 else "severe", **w))
    for ch in "AB":
        for suf, kn, g in (("", "normal", 0), ("_G2", "abnormal", 2), ("_G3", "severe", 3)):
            r = f"fit{ch}{suf}"
            try: L, fps = sim_L(f"results/ident/{r}.npz")
            except Exception as e: print("missing", r); continue
            for w in windows_of(angle_series(L, fps), fps): rows.append(dict(run=r, chain=ch, grade=g, knee=kn, **w))
    F = pd.DataFrame(rows); print(F.groupby(["chain", "knee"])[FEATS].median().round(0).to_string())
    base = F[F.chain == "base"]; A_, B_ = F[F.chain == "A"], F[F.chain == "B"]
    ORDER = ["normal", "abnormal", "severe"]; rng = np.random.default_rng(0)
    mk = lambda d, f: RandomForestClassifier(300, min_samples_leaf=2, random_state=0, class_weight="balanced").fit(d[f], d.knee)
    out = []
    base = base.assign(grp=base.run.str.replace(r"_g\d$", "", regex=True).str.replace(r"_G[23]", "_G", regex=True).str.replace(r"^(idg2?_)G\d", r"\1", regex=True))
    for nm, f in SETS.items():
        for sc in (0.0, 0.5, 1.0, 2.0):
            ps, ys = [], []
            for g in base.grp.unique():
                tr, te = base[base.grp != g], base[base.grp == g]
                if te.empty or tr.knee.nunique() < 3: continue
                ps += list(mk(tr, f).predict(noisy(te, f, sc, rng))); ys += list(te.knee)
            ps, ys = np.array(ps), np.array(ys)
            rec = {c: float((ps[ys == c] == c).mean()) for c in ORDER if (ys == c).any()}
            out.append(dict(test="base, leave-one-body-out", feats=nm, noise=sc, bal_acc=float(np.mean(list(rec.values()))), **{"recall_" + c: v for c, v in rec.items()}))
        for trn, tst, lb in ((base, pd.concat([A_, B_]), "base -> fitted chains A+B"), (pd.concat([base, A_]), B_, "base+A -> chain B"), (pd.concat([base, B_]), A_, "base+B -> chain A")):
            m = mk(trn, f)
            for sc in (0.0, 0.5, 1.0, 2.0):
                p = m.predict(noisy(tst, f, sc, rng)); rec = {c: float((p[tst.knee.values == c] == c).mean()) for c in ORDER if (tst.knee == c).any()}
                out.append(dict(test=lb, feats=nm, noise=sc, bal_acc=float(np.mean(list(rec.values()))), **{"recall_" + c: v for c, v in rec.items()}))
    T = pd.DataFrame(out); T.to_csv("results/ident_results/L_angles.csv", index=False)
    print(T[T.noise.isin([0.0, 1.0])].round(2).to_string())
    m = mk(pd.concat([base, A_, B_]), FEATS); imp = pd.Series(m.feature_importances_, FEATS).sort_values(ascending=False); imp.to_csv("results/ident_results/L_angles_importance.csv"); print(imp.round(2).to_string())
    z = np.load("data/real/leo2/work/leo_kp.npz", allow_pickle=True); kp, names = z["kp"], list(z["names"])
    VA = {k: angle_series(video_L(kp, names, int(a * 120), int(b * 120) - 1), 120) for k, (a, b) in {"A": (43.7, 46.8), "B": (33.8, 36.8)}.items()}
    LEO = {k: feats(A, 0, len(A["hock"])) for k, A in VA.items()}
    print(pd.DataFrame(LEO).round(0).to_string())
    o2 = []
    for lbl, tr in (("base", base), ("base+A+B", pd.concat([base, A_, B_]))):
        for nm, f in SETS.items():
            m = mk(tr, f)
            for off in (0, 6, 12, 18):
                for k, v in LEO.items():
                    x = dict(v); x["stif_mean"] += off; x["hock_mean"] += off
                    pr = dict(zip(m.classes_, m.predict_proba(pd.DataFrame([{c: x[c] for c in f}]))[0]))
                    o2.append(dict(train=lbl, feats=nm, offset=off, leo=k, p_normal=pr.get("normal", 0), p_abnormal=pr.get("abnormal", 0), p_severe=pr.get("severe", 0), call=max(pr, key=pr.get)))
    T2 = pd.DataFrame(o2); T2.to_csv("results/ident_results/L_angles_leo.csv", index=False); print(T2.round(2).to_string())
