import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.judge_knee import judge, load

rows = []
def add(name, truth, kp, names, a, b, fps, w, base=None):
    r = judge(kp, names, a, b, fps, w, base); rows.append(dict(video=name, truth=truth, weight=w, **r))
k2, n2 = load("data/real/leo2/work/leo_kp.npz")
for nm, (t0, t1) in {"leo2 A (walk)": (43.7, 46.8), "leo2 B (walk)": (33.8, 36.8)}.items(): add(nm, "normal", k2, n2, int(t0 * 120), int(t1 * 120) - 1, 120, 7.0)
SEG = {"n1": (0.0, 0.9), "n2": (1.1, 2.5), "n4": (0.4, 1.7), "n5": (0.6, 2.1)}
K3 = {t: load(f"data/real/leo3/work/{t}_kp.npz") for t in SEG}
for t, (t0, t1) in SEG.items():
    kp, nm = K3[t]; add(f"leo3 {t} (trot)", "normal", kp, nm, int(t0 * 120), int(t1 * 120) - 1, 120, 7.0)
    base = [(K3[o][0], K3[o][1], int(SEG[o][0] * 120), int(SEG[o][1] * 120) - 1, 120) for o in SEG if o != t]
    add(f"leo3 {t} (trot) vs other 3 baseline", "normal", kp, nm, int(t0 * 120), int(t1 * 120) - 1, 120, 7.0, base)
OD = [("d1_ttori_pre", "또리 수술 전", "abnormal(수술 필요)", 7, 30, 82, 126), ("d2_ttori_post", "또리 수술 후", "normal(수술 후)", 7, 60, 0, 89), ("d5_yanggaeng", "양갱", "normal에 가까움", 20, 60, 0, 89),
      ("d6_boksil", "복실", "normal에 가까움", 13, 60, 0, 119), ("d7_taepyeong_a", "태평 a", "severe(3기)", 13, 30, 0, 23), ("d8_taepyeong_b", "태평 b", "severe(3기)", 13, 30, 0, 23)]
for tag, nm, truth, w, fps, a, b in OD:
    z = np.load(f"data/real/dogs/cropwork/{tag}_kp.npz", allow_pickle=True); add(nm, truth, z["kp"], list(z["names"]), a, b, fps, w)
R = pd.DataFrame(rows); R.to_csv("results/judge_eval.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(R[["video", "truth", "call", "p_normal", "p_abnormal", "p_severe", "speed", "stif_mean", "reliable"]].round(2).to_string())
