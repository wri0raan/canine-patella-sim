import argparse, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from analysis.video_style_features import video_landmarks, features
from sklearn.ensemble import RandomForestClassifier

ROBUST = ["speed", "stride_hz", "step"]
PASSES = {"A": (43.7, 46.8), "B": (33.8, 36.8)}
FPS = 120
ORDER = ["normal", "abnormal", "severe"]
KO = {"normal": "정상", "abnormal": "이상", "severe": "심함"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mass", type=float, default=7.0); a = ap.parse_args()
    D = pd.read_csv("results/ident_results/dataset.csv"); gr = D[(D.hyp != "body") | D.cond.str.match(r"idg_M(10|08)F00_S")].copy()
    gr["knee"] = np.select([gr.grade == 0, gr.grade == 2], ["normal", "abnormal"], "severe")
    H = pd.read_csv("results/ident_results/H_knee3.csv"); h = H[(H.feats == "ROBUST") & (H.noise == 1.0)].iloc[0]
    C = pd.read_csv("results/ident_results/H_knee3_confusion.csv"); C = C[C.feats == "ROBUST"].set_index("knee")
    z = np.load("data/real/leo2/work/leo_kp.npz", allow_pickle=True); kp, names = z["kp"], list(z["names"])
    clf = RandomForestClassifier(500, min_samples_leaf=2, random_state=0, class_weight="balanced").fit(gr[ROBUST], gr.knee)
    rows = []
    for k, (t0, t1) in PASSES.items():
        f = features(video_landmarks(kp, names, int(t0 * FPS), int(t1 * FPS) - 1), FPS); x = pd.DataFrame([{c: f[c] for c in ROBUST}])
        pr = dict(zip(clf.classes_, clf.predict_proba(x)[0])); label = max(pr, key=pr.get)
        note = ("모델 상대값, 임상 진단 아님. 시뮬레이션 교차검증(잡음 1배, 3단계 균형 정확도 %.2f, 찍으면 0.33): 정상 %.0f %%, 이상 %.0f %%, 심함 %.0f %% 맞힘. '이상'은 대부분 '심함'으로 읽힘"
                % (h.bal_acc, 100 * h.recall_normal, 100 * h.recall_abnormal, 100 * h.recall_severe))
        if f["speed"] < 0.6: note += " 이 구간은 느려서(%.2f m/s) 학습 자료의 '이상' 걸음과 속도가 겹쳐 오판하기 쉬움 (속도 혼동)" % f["speed"]
        rec = dict(pass_=k, speed=f["speed"], stride_hz=f["stride_hz"], step=f["step"], mass_kg=a.mass, mass_input=True, knee=label, knee_ko=KO[label],
                   p_normal=float(pr.get("normal", 0)), p_abnormal=float(pr.get("abnormal", 0)), p_severe=float(pr.get("severe", 0)), note=note, cv_bal_acc_noise1=float(h.bal_acc))
        json.dump(rec, open(f"results/leo_estimates_{k}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1); rows.append(rec)
        print(k, {c: (round(v, 3) if isinstance(v, float) else v) for c, v in rec.items() if c != "note"})
    pd.DataFrame(rows).to_csv("results/leo_estimates.csv", index=False)


if __name__ == "__main__":
    main()
