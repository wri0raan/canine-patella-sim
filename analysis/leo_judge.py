import sys, json
import numpy as np
sys.path.insert(0, ".")
from analysis.leo_judge_lib import constants, clip_stifle, posterior

a = sys.argv[1:]; tags = a[0].split(","); tag = "+".join(tags); base = a[a.index("--baseline") + 1].split(",") if "--baseline" in a else []
import os; C = constants(runset=os.environ.get("RUNSET", "R3")); R_ = [clip_stifle(t, C["c_rump"]) for t in tags]; R_ = [r for r in R_ if np.isfinite(r[0])]; nt = len(R_)
if not nt: sys.exit("no frames with the whole dog in the picture: cannot judge")
x = float(np.mean([r[0] for r in R_])); win = float(sum(r[1] for r in R_)); legs = [float(np.mean([r[2][0] for r in R_])), float(np.mean([r[2][1] for r in R_]))]
notes = []
if win < 2.0: notes.append(f"short window ({win:.1f} s < 2 s, fewer than ~3 strides): the mean is unreliable")
if abs(legs[0] - legs[1]) > 12: notes.append(f"the two legs disagree by {abs(legs[0] - legs[1]):.0f} deg (tracking uncertain)")
if base:
    B = [clip_stifle(t, C["c_rump"])[0] for t in base]; mb = float(np.mean(B)); n = len(B); sdw = max(float(np.std(B, ddof=1)) if n > 1 else 4.5, 4.0)
    mus = {"normal": mb, "abnormal": mb + C["d_abnormal"], "severe": mb + C["d_severe"]}; sd = float(np.sqrt(sdw ** 2 * (1 / nt + 1 / n) + 2.0 ** 2)); mode = f"hybrid (baseline of {n} healthy clips: {mb:.1f} deg)"
else:
    mus = {"normal": C["mu_normal"], "abnormal": C["mu_normal"] + C["d_abnormal"], "severe": C["mu_normal"] + C["d_severe"]}
    sd = float(np.sqrt(4.5 ** 2 / nt + C["sd_sim"] ** 2 + C["sd_c"] ** 2 + C["sd_tracker"] ** 2)); mode = "absolute (no baseline)"
p = posterior(x, mus, sd); call = max(p, key=p.get); rel = p[call] >= 0.8
names = {"normal": "정상", "abnormal": "이상 (등급 2 수준)", "severe": "심함 (등급 3 수준)"}
print(f"[{tag}] mode: {mode}\n  stifle mean {x:.1f} deg (legs {legs[0]:.1f} / {legs[1]:.1f}), window {win:.1f} s\n  class means " + ", ".join(f"{names[k]} {v:.1f}" for k, v in mus.items()) + f"; sd {sd:.1f}")
print("  probability: " + ", ".join(f"{names[k]} {v:.2f}" for k, v in p.items())); print(f"  => {names[call]}" + ("" if rel else "  (판정 불가에 가까움: 확신도 < 0.8)"))
if not base and call in ("normal", "abnormal"): notes.append("absolute mode cannot reliably separate normal from grade 2 (8 deg apart vs ~6 deg sd); give --baseline healthy clips of this dog")
for n_ in notes: print("  note:", n_)
json.dump(dict(tag=tag, mode=mode, stifle_mean=x, legs=legs, window_s=win, class_means=mus, sd=sd, probability=p, call=call, reliable=bool(rel), notes=notes), open(f"results/judge_{tag}.json", "w"), indent=1, ensure_ascii=False)
