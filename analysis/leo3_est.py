import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.judge_knee import judge, load
KO = {"normal": "정상", "abnormal": "이상", "severe": "심함"}
k2, n2 = load("data/real/leo2/work/leo_kp.npz"); P = {}
for _, r in pd.read_csv("results/leo_segments.csv").iterrows():
    if r.speed_ms >= 2.0: P[f"leo2@{r.start:.0f}s"] = (k2, n2, int(r.start * 120), int(r.end * 120) - 1, 120)
SEG = {"n1": (0.0, 0.9), "n2": (1.1, 2.5), "n4": (0.4, 1.7), "n5": (0.6, 2.1)}
for t, (t0, t1) in SEG.items():
    kp, nm = load(f"data/real/leo3/work/{t}_kp.npz"); P[f"leo3 {t}"] = (kp, nm, int(t0 * 120), int(t1 * 120) - 1, 120)
for t in SEG:
    kp, nm, a, b, fps = P[f"leo3 {t}"]; base = [v for k, v in P.items() if k != f"leo3 {t}"]
    rb = judge(kp, nm, a, b, 120, 7.0, base, leo_model=True); ra = judge(kp, nm, a, b, 120, 7.0)
    est = dict(mass_kg=7.0, knee=rb["call"], knee_ko=KO[rb["call"]] + " (기준선 비교, 참고값)", p_normal=rb["p_normal"], p_abnormal=rb["p_abnormal"], p_severe=rb["p_severe"],
               note=rb["mode"] + ". 확신도 %.0f%%. 절대값 비교는 신뢰 낮음(%s). " % (100 * max(rb["p_normal"], rb["p_abnormal"], rb["p_severe"]), KO[ra["call"]]) + "속도가 모델 걷기 범위 밖이라 등급 변화량은 걷기에서 얻은 값.")
    json.dump(est, open(f"results/leo3_est_{t}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(t, rb["call"], round(rb["p_normal"], 2), round(rb["p_abnormal"], 2), round(rb["p_severe"], 2))
json.dump(dict(mass_kg=7.0, knee=None, note="분석 불가: 개가 화면에 잘리고 뒷모습이어서 걸음 구간이 없음"), open("results/leo3_est_n3.json", "w", encoding="utf-8"), ensure_ascii=False)
