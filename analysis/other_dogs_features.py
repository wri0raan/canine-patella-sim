import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.video_style_features import video_landmarks, features
from analysis.leo_segments import smooth_series

DOGS = {
    "d1_ttori_pre": ("또리 수술 전", 7, "수술 전(슬개골 탈구)", 30), "d2_ttori_post": ("또리 수술 후", 7, "수술 후", 60),
    "d3_gamja_a": ("감자 a", 10, "정상", 60), "d4_gamja_b": ("감자 b", 10, "정상", 60), "d5_yanggaeng": ("양갱", 20, "정상에 가까움(모름)", 60),
    "d6_boksil": ("복실", 13, "정상에 가까움(모름)", 60), "d7_taepyeong_a": ("태평 a", 13, "3기", 30), "d8_taepyeong_b": ("태평 b", 13, "3기", 30)}
WIN_S, HOP_S = 1.5, 0.25
rows = []
for tag, (nm, kg, st, fps) in DOGS.items():
    z = np.load(f"data/real/dogs/cropwork/{tag}_kp.npz", allow_pickle=True); kp, names = z["kp"], list(z["names"]); ix = {n: i for i, n in enumerate(names)}; T = len(kp)
    P = lambda n: smooth_series(kp[:, ix[n], :2], kp[:, ix[n], 2], thr=0.3, win=9); C = lambda n: kp[:, ix[n], 2]
    body = np.linalg.norm(P("back_base") - P("back_end"), axis=1) + 1e-6
    head_down = ((P("nose")[:, 1] - P("neck_base")[:, 1]) / body) > 0.25
    conf = {s: np.mean([np.median(C(f"back_{s}_{k}")) for k in ("thai", "knee", "paw")]) for s in ("left", "right")}
    w = int(WIN_S * fps); h = max(1, int(HOP_S * fps)); best = None
    for a in range(0, T - w + 1, h):
        b = a + w - 1
        if head_down[a:b + 1].mean() > 0.3: continue
        sc = np.mean([np.median(C(f"{leg}_{k}")[a:b + 1]) for leg in ("back_left", "back_right") for k in ("thai", "knee", "paw")])
        if best is None or sc > best[0]: best = (sc, a, b)
    rec = dict(tag=tag, dog=nm, kg=kg, status=st, fps=fps, frames=T, head_down_frac=float(head_down.mean()), hind_conf=max(conf.values()))
    if best is None: rec["note"] = "걷는 구간 없음(고개를 숙이거나 너무 짧음)"
    else:
        try:
            f = features(video_landmarks(kp, names, best[1], best[2]), fps)
            rec.update(win_start_s=best[1] / fps, win_conf=best[0], stride_hz=f["stride_hz"], hock_min=f["hock_min"], hock_max=f["hock_max"], hock_rom=f["hock_max"] - f["hock_min"],
                       carpus_min=f["carpus_min"], carpus_max=f["carpus_max"], carpus_rom=f["carpus_max"] - f["carpus_min"], bounce=f["bounce"])
        except Exception as e: rec["note"] = f"특징 계산 실패: {e}"
    rows.append(rec)
R = pd.DataFrame(rows); R.to_csv("results/other_dogs_features.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(R.drop(columns=["status", "dog"]).round(2).to_string())
