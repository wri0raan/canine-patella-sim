<div align="center">

# 🐕 Canine Patella Sim

**Video-calibrated physics model of a dog, and what patellar luxation does to its gait**

Undergraduate thesis · CUCG Lab, The Catholic University of Korea · 2026

<img src="docs/readme/healthy_vs_grade3.gif" width="760" alt="Healthy gait vs. Grade III patellar luxation" />

<sub>Left: healthy model fitted to Leo · Right: same body with a Grade III knee-extension limit</sub>

[Project page](https://wri0raan.github.io/JURAN/research/gait.html) · [Portfolio](https://wri0raan.github.io/JURAN/)

</div>

---

## The question

Small dogs with patellar luxation can't fully straighten the knee, and in the clinic they walk with extra motion at the hip and hock to make up for it.

**Is the knee limit alone enough to produce that compensation?**
This project tests it in physics: build a dog model from video of a real dog, lock its knee to clinically measured limits per grade, re-optimize the gait, and compare the joint angles with clinical measurements.

## Pipeline

<table>
<tr>
<td width="50%"><img src="docs/readme/keypoints.jpg" alt="Keypoints from video" /></td>
<td width="50%"><img src="docs/readme/skin_skeleton.gif" alt="Skin and skeleton render" /></td>
</tr>
<tr>
<td align="center"><sub>Keypoints from 4K 120 fps video (SuperAnimal)</sub></td>
<td align="center"><sub>Fitted body, rendered with skin and skeleton</sub></td>
</tr>
</table>

**A. Video → healthy model**

1. **Film** — Leo (7 kg mixed breed), side view, 4K 120 fps; walking passes picked from contact sheets
2. **Keypoints** — SuperAnimal-Quadruped (DeepLabCut) → stride period, fore/hind order, swing height
3. **Body** — measured segment lengths + a standing side photo → per-segment scale of the MuJoCo / dm_control Dog
4. **Healthy gait** — CMA-ES, adding cost terms one stage at a time; validated on video metrics that were *not* used in the fit
   → simulated stride period **0.43 s** vs. Leo **0.41 s**

**B. Healthy model → clinical grades**

5. **Inject the grade** — knee-extension limit per grade as a joint limit (mechanical / functional)
6. **Re-optimize under three hypotheses** — H1 minimize energy · H2 keep healthy joint excursions · H3 avoid load (pain); plus a one-sided limit; 3 seeds each
7. **Compare** — knee, hock and hip angles against clinical measurements (Chayatup et al., *Vet J* 313, 2025)
8. **Render** — skin + skeleton comparison videos per grade

<div align="center">
<img src="docs/readme/grades_skeleton.gif" width="760" alt="Skeleton gait across grades" />
<br><sub>Healthy · Grade 2 · Grade 3 · Grade 4</sub>
</div>

## Results

<img src="docs/readme/final_grades_leo.png" alt="Simulated vs clinical joint angles by grade" />

- **Knee** — the Grade III knee angle lands in the clinical range **only under H2** (keep healthy excursions).
- **Hock** — clinical dogs increase hock motion with grade; **no hypothesis and no model variant reproduced it.**
- **Takeaway** — the knee limit alone does not explain compensatory gait. Something beyond the joint constraint (behavior, muscle, pain) is needed.

A mismatch with clinical data is treated as a result, not a failure.

## Next

Turn this into a tool that builds a per-dog physics model from video:
a second dog to test generality → automate the fitting → add a muscle layer and a learned controller → validate on videos of real patients.

## Repository layout

```
sim/        MuJoCo / dm_control dog: scaling, gait, impairment, rollout
configs/    gait, impairment (patella.yaml) and model (leo.yaml) settings — no magic numbers in code
scripts/    pose extraction, optimization runners (run_*.sh), rendering
analysis/   metrics shared by sim and video, figures, clinical comparison
docs/       PIPELINE.md (step-by-step reproduction), research notes
```

## Reproduce

Three environments: `.venv` (simulation: MuJoCo, dm_control, CMA-ES), `.venv_dlc` (pose estimation: DeepLabCut, PyTorch, GPU), `.venv_bpy` (Blender rendering).
Every stage, its command and its outputs are listed in **[docs/PIPELINE.md](docs/PIPELINE.md)**.

```bash
sh scripts/run_leo_chain4.sh        # healthy gait, staged
python scripts/eval_leo_gait.py leoH3   # independent check against video
sh scripts/run_leo_sweeps.sh        # grades × hypotheses × seeds
python analysis/final_results.py --leo --prefix s   # tables and figures
```

## Data

Leo's videos and X-rays are not included. Reference videos of other dogs were used for qualitative comparison only and are not redistributed.
Clinical values come from Chayatup, Wangdee & Soontornvipart, *The Veterinary Journal* 313 (2025) 106369 — only Grade 0 and Grade III are measured there; other grades are interpolated and labelled as such.

---

<div align="center">
<sub>Juran Kim · <a href="https://wri0raan.github.io/JURAN/">wri0raan.github.io/JURAN</a> · with Leo 🐾</sub>
</div>
