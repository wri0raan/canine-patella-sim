from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
sweeps = [f"{c}{s}" for s in (1, 2, 3) for c in ("swF2d", "swM1d", "swF1d")] + ["swH3d1", "swU2d1", "sensH2", "sensH1"]
print("== sweeps (summary csv = finished) ==")
for t in sweeps:
    done = (R / f"{t}_summary.csv").exists()
    parts = sorted(p.name for p in R.glob(f"{t}_g*") if (p / "done.txt").exists() or (p / "best.npz").exists())
    fin = sorted(p.name for p in R.glob(f"{t}_g*") if (p / "done.txt").exists())
    print(f"  {t:8s} {'DONE' if done else 'todo':5s} grades finished: {len(fin)}  dirs: {len(parts)}")
for f in ("results_queue_progress.txt", "results_final_sweeps_progress.txt", "results_extra_sweeps_progress.txt"):
    p = ROOT / f
    if p.exists():
        print(f"== {f} ==\n" + p.read_text(encoding="utf-8").strip())
print("\nResume the queue (skips finished sweeps/grades):  sh scripts/run_queue.sh")
