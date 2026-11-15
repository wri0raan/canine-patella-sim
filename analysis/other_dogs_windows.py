import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
from analysis.video_style_features import video_landmarks, features

W = [("d1_ttori_pre", "또리 수술 전", 7, "수술 전", 30, 82, 126, "walk"), ("d2_ttori_post", "또리 수술 후", 7, "수술 후", 60, 0, 89, "walk"),
     ("d5_yanggaeng", "양갱", 20, "정상에 가까움", 60, 0, 89, "walk"), ("d6_boksil", "복실", 13, "정상에 가까움", 60, 0, 119, "walk"),
     ("d7_taepyeong_a", "태평 a", 13, "3기", 30, 0, 23, "walk"), ("d8_taepyeong_b", "태평 b", 13, "3기", 30, 0, 23, "walk"),
     ("d3_gamja_a", "감자 a(냄새)", 10, "정상", 60, 0, 89, "sniff"), ("d4_gamja_b", "감자 b(냄새)", 10, "정상", 60, 0, 89, "sniff")]
rows = []
for tag, nm, kg, st, fps, a, b, kind in W:
    z = np.load(f"data/real/dogs/cropwork/{tag}_kp.npz", allow_pickle=True); kp, names = z["kp"], list(z["names"])
    f = features(video_landmarks(kp, names, a, b), fps)
    rows.append(dict(tag=tag, dog=nm, kg=kg, status=st, kind=kind, fps=fps, seconds=(b - a + 1) / fps, stride_hz=f["stride_hz"], hock_rom=f["hock_max"] - f["hock_min"], hock_min=f["hock_min"], hock_max=f["hock_max"],
                     carpus_rom=f["carpus_max"] - f["carpus_min"], carpus_min=f["carpus_min"], carpus_max=f["carpus_max"], bounce=f["bounce"]))
z = np.load("data/real/leo2/work/leo_kp.npz", allow_pickle=True); kp, names = z["kp"], list(z["names"])
for nm, t0, t1 in (("leo A", 43.7, 46.8), ("leo B", 33.8, 36.8)):
    f = features(video_landmarks(kp, names, int(t0 * 120), int(t1 * 120) - 1), 120)
    rows.append(dict(tag=nm, dog=nm, kg=7, status="정상(참고)", kind="walk", fps=120, seconds=t1 - t0, stride_hz=f["stride_hz"], hock_rom=f["hock_max"] - f["hock_min"], hock_min=f["hock_min"], hock_max=f["hock_max"],
                     carpus_rom=f["carpus_max"] - f["carpus_min"], carpus_min=f["carpus_min"], carpus_max=f["carpus_max"], bounce=f["bounce"]))
R = pd.DataFrame(rows); R.to_csv("results/other_dogs_windows.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(R.drop(columns=["tag", "status"]).round(2).to_string())
col = {"정상(참고)": "#1f77b4", "정상": "#1f77b4", "정상에 가까움": "#2ca02c", "수술 전": "#ff7f0e", "수술 후": "#9467bd", "3기": "#d62728"}
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
for ykey, yl, a_ in (("hock_rom", "비절 각도 범위 (도, 2D)", ax[0]), ("carpus_rom", "손목 각도 범위 (도, 2D)", ax[1]), ("stride_hz", "걸음 주기 (Hz)", ax[2])):
    for i, r in R.iterrows():
        if np.isfinite(r[ykey]): a_.scatter(r.kg + (0.15 if r.kind == "sniff" else 0), r[ykey], c=col[r.status], marker="x" if r.kind == "sniff" else "o", s=70); a_.annotate(r.dog, (r.kg, r[ykey]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a_.set_xlabel("체중 (kg)"); a_.set_ylabel(yl); a_.grid(alpha=.3)
fig.suptitle("다른 개와 leo의 걸음 특징 (속도는 측정 불가) - 색: 슬개골 상태, x: 냄새 맡으며 걷기", fontsize=10); fig.tight_layout(); fig.savefig("results/figures/other_dogs_windows.png", dpi=150)
