import argparse, os, subprocess, sys
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("tag"); ap.add_argument("--start", type=float); ap.add_argument("--dur", type=float)
ap.add_argument("--stages", default="crop,pose,convert,tracks,judge"); ap.add_argument("--baseline", default=""); a = ap.parse_args()
PYD, PY = os.path.abspath(".venv_dlc/Scripts/python.exe"), os.path.abspath(".venv/Scripts/python.exe"); env = dict(os.environ, PYTHONPATH=".", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"); st = a.stages.split(","); src = a.video
def run(cmd): print("$", " ".join(cmd), flush=True); r = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL); r.returncode and sys.exit(f"stage failed: {cmd[1]}")
if a.start is not None:
    import imageio_ffmpeg; src = f"data/real/leo5/{a.tag}.mp4"
    run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-ss", str(a.start), "-t", str(a.dur or 8), "-i", a.video, "-an", "-r", "120", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", src])
os.makedirs("data/real/leo5/work", exist_ok=True)
if "crop" in st: run([PYD, "scripts/leo5_crop.py", src, f"data/real/leo5/work/c{a.tag}"])
if "pose" in st: run([PYD, "scripts/leo5_pose.py", a.tag])
if "convert" in st: run([PYD, "scripts/leo5_convert.py"])
if "tracks" in st: run([PY, "analysis/leo5_tracks2.py", a.tag])
if "judge" in st: run([PY, "analysis/leo_judge.py", a.tag] + (["--baseline", a.baseline] if a.baseline else []))
