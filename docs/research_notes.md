# Research notes (2026-10-17)

Status legend: [verified] = read from a source this session / run locally, [unverified] = from memory or a prior chat, check before citing.

## 1. Model candidates

### dm_control Dog (chosen primary) [verified locally]
- Source: `dm_control/suite/dog.xml`, and `locomotion/walkers/assets/dog_v2/`. dm_control paper (arXiv 2006.12983): Pharaoh Hound model by leo3Dmodels, kinematics/skinning/collision generated procedurally with PyMJCF. Muscle/tendon attachment points exist only in the static model; muscle-driven dynamics is "future work" in the paper.
- Local dump (`results/dog_dump_suite_*.csv`): nbody=62, njnt=74, nu=38, **total mass 10.24 kg** (close to Leo's 7 kg -> length scale about (7/10.24)^(1/3) = 0.88), timestep 0.005.
- Hind limb joints (per side): `hip_*_supinate` [-20,20], `hip_*_abduct` [-30,90], `hip_*_extend` [-130,60], `knee_*` [-120,10] (axis -y), `ankle_*` [-100,20] (this is the hock / tarsus), `toe_*` [-15,70]. Spine L_1..L_7, neck C_1..C_7, tail Ca_1..Ca_21 are passive joints with stiffness; spine/neck/tail are driven by 8 tendon actuators.
- Actuators: 30 joint torque motors (no built-in PD; gear: hip_extend 40, knee 30, ankle 20, toe 2 ...), ctrlrange [-1, 1] after rescale, plus 8 tendon actuators. We must build our own PD (plan: convert to MuJoCo position servos with implicit damping, torque cap = gear).
- Suite `Stand` task randomizes root heading, **root velocity (up to several m/s)** and actuator state at reset -> do not use `env.reset()` state as a standing pose.
- Knee convention: range [-120, 10] deg, q=0 is about straight. Clinical angle (femur-tibia included angle, 180 = straight) mapping must be verified by forward kinematics before comparing with the clinical table.
- No OpenSim/MuJoCo dog with open files found elsewhere (see below), no dog in MuJoCo Menagerie, LocoMuJoCo has Unitree robots only.

### Canine OpenSim hindlimb model (reference, not a dependency)
- Brown et al. 2020, Frontiers Bioeng Biotech, 5.4 kg Dachshund bilateral pelvic-limb model: hip 3 DOF, stifle 3 DOF (+2 coupled translations), tarsus 1 DOF; data "available on request"; open-access paper (CC BY). Model files not confirmed public. Reported model walking ROM (stance/swing): hip 30/36 deg, stifle 13/37, tarsus 28/40; they note mismatch with published trotting values. https://pmc.ncbi.nlm.nih.gov/articles/PMC7079575/
- Other: multi-body canine hind limb with GRF (BioMed Eng Online 2009/2011): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2787502/ ; U. Louisville thesis (canine pelvic limb neuromusculoskeletal): https://ir.library.louisville.edu/etd/2367

### Unity project in this machine (not used for physics)
- `PawsWatch AI/Assets/SIGGRAPH_2022/...` = Starke et al. DeepPhase supplementary: kinematic/learned, has German Shepherd mesh+fur+rig and 51 dog mocap clips (`D1_*_KAN01_*.bvh.asset`). No license file found. Possible later use: visualization only. [verified listing]

## 2. Clinical data

- Notion table (user): AROM at trot 1.11 m/s per Grade, PROM per Grade. Rows marked as measured (yellow) are **Grade 0 and Grade 3 only**; Grade 1, 2, 4 look interpolated (rounded values, "about", "+"). AROM max extension at Grade 0 (167.1) exceeds PROM max extension (162.9): different protocols, do not mix.
- **Probable source: Chayatup K., Chulalongkorn Univ. thesis (2016), "A 2D kinematic comparison of hind limb gait during treadmill trotting in Chihuahuas with normal stifles versus medial patellar luxation stifles treated surgically."** Trot 1.11 m/s, grade III MPL: stifle ROM reduced, hip and tarsal ROM increased (p<0.001), normalized 8 weeks after surgery. The earlier assumption "Pomeranian" is likely wrong (Chihuahua is likely the study dog; **needs the user to confirm which PDF the Grade table came from**). Full text needs a Chula login: https://digiverse.chula.ac.th/Info/item/dc:54618
- Context: toy poodle standing CT grade 2 (abnormal posture, no bone deformity) https://pure.nihon-u.ac.jp/en/publications/evaluation-of-hindlimb-deformity-and-posture-in-dogs-with-grade-2/ ; French bulldog X-ray gait: patella more distal/medial in MPL https://pmc.ncbi.nlm.nih.gov/articles/PMC7881552 ; FE model of femoral varus MPL https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7713038/
- A Chihuahua is about 2-3 kg: scaling to a 10 kg model/7 kg Leo is by joint angle (scale-free) and Froude-scaled speed; need the study dogs' limb length.

## 3. Related work (novelty positioning)

- Robot fault-tolerant locomotion (impaired/locked joints, RL): Saving the Limping https://arxiv.org/abs/2210.00474 ; Learning Quadrupedal Locomotion with Impaired Joints (random joint masking) https://arxiv.org/abs/2403.00398 ; AcL Action Learner (2025, Go2) https://arxiv.org/abs/2503.21401 ; Contrastive forward prediction RL (CoRL 2025). => robot side is not novel; use only as generalization experiment.
- Human predictive simulation of pathological gait with contracture/weakness (emergent compensation): Bidirectional GaitNet (contracture as ROM limit) https://arxiv.org/abs/2306.04161 ; Falisse et al. 2020 cerebral palsy (Front Hum Neurosci) ; toe-walking contracture simulation. => "emergent compensation from constraints" exists for humans; **the gap I could not find a counterexample for is a canine, clinically-graded (patellar luxation) version validated on measured ROM**. Only 1-2 searches per topic, so do a proper literature pass (Google Scholar: "canine gait predictive simulation", "OpenSim dog") before claiming novelty.
- Physics character animation context: Locomotion Skills for Simulated Quadrupeds (SIGGRAPH 2011), DeepMimic (2018), Peng et al. imitating animals (RSS 2020). [unverified details]
- SIGGRAPH 2027 Posters: official CFP not found yet (third-party tracker lists TBD). SIGGRAPH Asia 2026 posters closed 2026-07-31.

## 4. Update 2026-10-17 (after reading the primary source)

**Primary clinical source (full text read, `data/clinical/06/`)**: Chayatup, Wangdee, Soontornvipart, *Preoperative and postoperative joint motion in chihuahuas with Grade III medial patellar luxation: a kinematic and goniometric analysis*, The Veterinary Journal 313 (2025) 106369, https://doi.org/10.1016/j.tvjl.2025.106369
- 7 healthy Chihuahuas (3.12 kg) vs 10 Grade III MPL (3.25 kg; 8 unilateral, 2 bilateral with contralateral Grades I/II). Treadmill trot 1.11 m/s (mean Chihuahua trot speed), 2D sagittal, 5 markers, 120 fps, 8 Hz Butterworth, 45 strides per dog. Joint angle = arccos of adjacent segment vectors (= included angle, same as ours). Hip angle uses iliac crest-greater trochanter vs femur.
- Table 2 (active, trot): stifle AROM 48.93 -> 28.77, MEA 167.1 -> 125.1, MFA 121.8 -> 96.9 (MPL flexes MORE = crouch); hip AROM 19.0 -> 35.5, MEA 142.1 -> 143.9, MFA 122.9 -> 108.4; tarsal AROM 47.6 -> 78.4, MEA 157.4 -> 168.7, MFA 105.8 -> 89.2.
- Table 4 (passive): stifle PROM 117.3 -> 103.9, MEA 162.9 -> 149.7, MFA 44.7 -> 45.8; hip unchanged; tarsal PROM 131.4 -> 125.3.
- Post-op weeks 2/4/6/8 give partial-recovery states (Tables 3, 5, supplement) = extra intermediate severity points for validation.
- **Only Grade 0 and Grade 3 are measured**. Grade 1, 2, 4 in the Notion table are not in this paper (interpolated). The Notion AROM numbers for grade 0 differ slightly from the paper (likely the 2016 thesis).
- Validation targets are therefore MEA/MFA per joint (not only ROM): the emergent compensation to test is stifle window [121.8,167.1] -> [96.9,125.1], tarsal MEA up, hip MFA down.
- Dynamic similarity: Chihuahua 3.12 kg at 1.11 m/s -> our 10.24 kg Dog: length ratio (10.24/3.12)^(1/3) = 1.485, equal Froude number -> speed x sqrt(1.485) = **1.35 m/s** (my earlier 1.5 m/s was a hip-height guess).

**Other PDFs from the Notion page** (downloaded with permission to `data/clinical/NN/`): 03 = Ahn et al. 2026, underwater treadmill joint angles pre/post surgery in Pomeranians (J Environ Sci Int; not the source of the grade table; short communication, flexion/extension angles). 06 = the paper above. 01, 02, 04, 05, 07 not downloaded yet (literature review, diagnosis/treatment, deep-learning detection, bulldog radiographs, FE model).

**Additional literature found (search only, not read)**
- Chayatup 2016 Chulalongkorn thesis; "A comparative study of ROM of forelimb and hind limb in walk and trot pattern of Chihuahuas affected/non-affected with PL" (ResearchGate 282522553): similar group, MPL vs normal.
- Muscle-driven predictive physics simulation of equine locomotion (2024), https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11428545/ : closest animal predictive-simulation work; no lameness/joint restriction.
- Falisse et al. 2020, Front Hum Neurosci 14:40, https://doi.org/10.3389/fnhum.2020.00040 (code on SimTK "predictcpgait"): predictive simulation, musculoskeletal vs motor-control deficits -> crouch gait; supports the framing "which deficit explains the pattern" (our mechanical vs functional vs behavioral).
- Kinematics of canine gait by breed: Fischer & Lilje (32 breeds), Labrador stifle/tarsus angles (CCL predisposition), Greyhound familiarisation study, Beagle inverse dynamics.
- Novelty statement still unverified: no study found that reproduces canine MPL compensation by predictive simulation, but only ~8 searches were done.

## 5. Literature pass for novelty / SIGGRAPH poster (2026-10-17, search snippets only, full texts NOT read)
- No study found that simulates whole-limb canine patellar-luxation gait with joint constraints and compares hip/tarsal compensation to measured kinematics. Closest: Lee et al. 2020 BMC Vet Res (finite-element model of MPL with femoral varus, patellofemoral joint only, https://link.springer.com/article/10.1186/s12917-020-02644-5); Brown, Bertocci, States, Levine, Levine, Howland, Front Bioeng Biotechnol 8:150 (2020), canine hindlimb musculoskeletal model for walking gait (no luxation); Chayatup 2016 / 2025 (measured kinematics).
- Physics-based quadruped animation: Coros, Karpathy, Jones, Reveret, van de Panne, "Locomotion skills for simulated quadrupeds", SIGGRAPH/ACM TOG 2011 (https://dl.acm.org/doi/10.1145/2010324.1964954): simulated dog, many gaits, no injury or constraint-induced adaptation.
- Human predictive simulation of crouch/pathological gait: Falisse et al. 2020; Gait & Posture 2025 muscle-synergy crouch simulation (snippet only). Caution reported in the literature: 2D predictive simulations can be less natural; simulated ROMs can exceed measured ones (crutch gait example).
- Robot fault-tolerant locomotion papers exist (see section 3); they concern robots, not clinical grades.
- SIGGRAPH 2027 Posters: no call found yet (third-party tracker says TBD). Posters are for new, unpublished work, and work under review elsewhere cannot be submitted -> **check the prior-publication rule before the thesis becomes public** (school repository timing). SIGGRAPH Asia 2026 posters closed 2026-07-31, Asia 2025 deadline was 2025-08-20 (pattern: late July to August).
- Statement that can safely be made today: "to the best of our (limited) search, no prior clinically-graded physics simulation of canine patellar luxation gait validated against measured joint angles was found". Needs a fuller literature search before submission.
