import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.identify_angles import video_L, angle_series, feats

W = [("d1_ttori_pre", "또리 수술 전", "수술 전", 30, 82, 126), ("d2_ttori_post", "또리 수술 후", "수술 후", 60, 0, 89), ("d5_yanggaeng", "양갱", "정상에 가까움", 60, 0, 89),
     ("d6_boksil", "복실", "정상에 가까움", 60, 0, 119), ("d7_taepyeong_a", "태평 a", "3기", 30, 0, 23), ("d8_taepyeong_b", "태평 b", "3기", 30, 0, 23),
     ("d3_gamja_a", "감자 a(냄새)", "정상(냄새)", 60, 0, 89), ("d4_gamja_b", "감자 b(냄새)", "정상(냄새)", 60, 0, 89)]
rows = []
for tag, nm, st, fps, a, b in W:
    z = np.load(f"data/real/dogs/cropwork/{tag}_kp.npz", allow_pickle=True)
    try: A = angle_series(video_L(z["kp"], list(z["names"]), a, b), fps)
    except Exception as e: print("fail", tag, e); continue
    rows.append(dict(dog=nm, status=st, fps=fps, seconds=round((b - a + 1) / fps, 1), **feats(A, 0, len(A["hock"]))))
z = np.load("data/real/leo2/work/leo_kp.npz", allow_pickle=True)
for nm, t0, t1 in (("leo A", 43.7, 46.8), ("leo B", 33.8, 36.8)):
    A = angle_series(video_L(z["kp"], list(z["names"]), int(t0 * 120), int(t1 * 120) - 1), 120)
    rows.append(dict(dog=nm, status="정상(참고)", fps=120, seconds=round(t1 - t0, 1), **feats(A, 0, len(A["hock"]))))
R = pd.DataFrame(rows); R.to_csv("results/other_dogs_angles.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(R.round(0).to_string())
