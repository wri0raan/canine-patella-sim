# Direction 3 roadmap: "video -> physics twin of an animal" (graphics / robotics)  (2026-10-20)

User choice (2026-10-20): the third direction of the thesis outlook = a tool that builds an individual animal's physics model from video, then synthesises (patho)logical gait and renders it.

## What exists (thesis-level evidence)
- Video analysis: keypoints (SuperAnimal), pass selection, gait metrics (stride period, fore-hind lag, swing clearance, footprint line).
- Body fit: hand-tuned region scales validated by photo overlay; **prototype auto-fit** from standing photos (analysis/auto_body_fit.py): median over 10 photo frames, tibia +2 %, trunk +1 %, metatarsus -12 %, forearm -20 %, fore paw -12 %, neck+head -32 % (landmark definition), tail unusable (curled plume); mean 13 %.
- Physics: MuJoCo model with skin, CMA-ES gait synthesis with video-derived cost terms (staged), independent validation report.
- Impairment injection (clinical joint limits) and hypothesis tests; classifier pilot (grade 0.87 at 0 deg noise, cause 0.60; limit regression MAE 1.2-5.5 deg).
- Rendering: skeleton + skin videos; Border Collie re-skin by bone retargeting (limbs absolute, spine delta), ear and coat variants.

## Gaps for a graphics/robotics contribution (ordered by value / effort)
1. **Second animal** (generality): run the whole pipeline on another dog (own videos or consented clips). Without it the claim is "one case study". 1-2 weeks.
2. **One-command automation**: wrap stages 2-8 (`pipeline.py video.mp4 --mass 7 --height 0.35 ...`): auto pass selection (partly in leo_segments), auto body fit (extend to pose-invariant segment lengths + several photos), staged calibration chain with automatic stage advancement and acceptance gates. 1-2 weeks.
3. **Fidelity metrics**: report a fixed table (period, lag, clearance, footprint line, speed, hock/stifle windows) for every animal; add a held-out video clip as test.
4. **Muscle / fat layers** (user idea): torque-limit scaling (weakness/strength) and mass/girth (fat) as parameters; also tests the missing-factor hypothesis for the hock compensation. 1-2 weeks.
5. **Rendering quality**: fit the skin to the animal (shape morph or Gaussian-splat / photogrammetry from the video), fur shading (Eevee/Cycles), ear/tail secondary motion. 2-4 weeks.
6. **Control quality**: closed-loop or learned controller (RL/imitation, DeepMimic-style) to replace open-loop sinusoids; spine/scapula motion. 4+ weeks.
7. **Validation with real impaired dogs** (vet-labelled videos): data dependent; keep as a collaboration goal.

## Venues (checked by web search on 2026-10-19/20)
- SIGGRAPH Asia 2026 Posters: deadline 31 Jul 2026 23:59 AoE, **closed**; conference 1-4 Dec 2026, Kuala Lumpur.
- SIGGRAPH 2027: poster call and deadline not found yet (a third-party tracker lists the paper deadline as TBD, conference Aug 2027). Check the official site when it opens; also SIGGRAPH Asia 2027 posters (earlier years: late July / mid-August deadlines).
- Alternatives to check: ACM SCA / MIG (animation), workshops on animal pose at CVPR/ECCV, ICRA/IROS (legged robots), a veterinary journal for the clinical side. Posters must be unpublished and short (2 pages); a public thesis or repository may count as prior work, so read each call's policy first.

## Claims to avoid
- "digital twin" of Leo (shape is a scaled generic dog, the appearance is a re-skin), real diagnosis from video (no labelled data), hock compensation explained (it is not).

## Suggested next 3 steps after the thesis
1. Acquire a second animal video; run stages 2-6 and report the fidelity table. 2. Implement the muscle/fat parameters and rerun H2 grade 3. 3. Draft a 2-page poster: pipeline figure + fidelity table + skin comparison video frames.
