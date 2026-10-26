"""Evaluate a BeakPick ONNX policy on the candidate objects in CPU MuJoCo (BAM servos).

For each object: stand 1 s, run one pick cycle (5 s), stand 1 s more. Success = the
object ends > 30 mm above its resting height AND within 30 mm of the beak grasp point,
with the robot standing (trunk > 90 mm; a deep crouch legitimately dips below 70 mm).
Placement per trial: nominal spot + uniform noise (training range) and random yaw.

usage: python scripts/pick_eval.py <pick.onnx> [trials=5] [keys=all|k1,k2] [video_dir|-] [seed=0]
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import mujoco

from microduck_tidy import sim as S, pick_profile as P
from microduck_tidy.props import CATALOG, GRASPABLE, BORDERLINE
from microduck_tidy.runner import PickRunner

ONNX = sys.argv[1]
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
KEYS = ([p.key for p in GRASPABLE + BORDERLINE] if len(sys.argv) <= 3 or sys.argv[3] == "all"
        else sys.argv[3].split(","))
VIDEO_DIR = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
if VIDEO_DIR:
    os.makedirs(VIDEO_DIR, exist_ok=True)
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
rng = np.random.default_rng(SEED)
TRACE = os.environ.get("TRACE") == "1"


def trial(spec, ox, oy, yaw, video=None, trace=False):
    xy = (P.NOMINAL[0] + ox, P.NOMINAL[1] + oy)
    sim = S.build(S.SCENE_FLAT, catalog=[("target", spec, xy, yaw)])
    run = PickRunner(sim, ONNX)
    run.reset_standing()
    rest_z = run.prop_pos("target")[2]
    writer = cam = r = None
    if video:
        import imageio
        r = mujoco.Renderer(sim.model, 480, 640)
        cam = mujoco.MjvCamera(); cam.distance = 0.42; cam.azimuth = 135; cam.elevation = -18
        writer = imageio.get_writer(video, fps=25, codec="libx264", macro_block_size=1)
    max_z, fell = rest_z, False
    for k in range(int(7.0 * 50)):
        if k == 50:
            run.start_pick("target")
        run.step()
        z = run.prop_pos("target")[2]
        max_z = max(max_z, z)
        if trace and k % 12 == 0:
            ph = "-" if run.phase is None else f"{run.phase:.2f}"
            print(f"    t={k/50:4.2f} phase={ph:>4s} trunk_z={run.trunk_z()*1000:3.0f}mm prop_z={z*1000:5.1f}mm "
                  f"grasp_d={run.grasp_dist('target')*1000:5.1f}mm mouth={sim.beak.angle:.2f} "
                  f"offset_obs=({run.offset_of('target')[0]*1000:+.0f},{run.offset_of('target')[1]*1000:+.0f})mm")
        if writer is not None and k % 2 == 0:
            x0, y0, _ = run.root_xy_yaw()
            cam.lookat[:] = [x0 + 0.05, y0, 0.09]
            r.update_scene(sim.data, camera=cam); writer.append_data(r.render())
    if writer is not None:
        writer.close()
    fell = run.trunk_z() < 0.09
    lift = run.prop_pos("target")[2] - rest_z
    held = run.grasp_dist("target")
    ok = lift > 0.03 and held < 0.03 and not fell
    return ok, lift, held, fell, max_z - rest_z


print(f"policy: {ONNX}")
print(f"{'object':22s} {'ok':>5s}  lift_mm(mean)  held_mm(mean)  falls")
summary = []
for key in KEYS:
    spec = CATALOG[key]
    res = []
    for t in range(TRIALS):
        ox = rng.uniform(-P.NOISE_X, P.NOISE_X)
        oy = rng.uniform(-P.NOISE_Y, P.NOISE_Y)
        yaw = rng.uniform(-0.6, 0.6)
        vid = os.path.join(VIDEO_DIR, f"{key}_{t}.mp4") if VIDEO_DIR and t == 0 else None
        res.append(trial(spec, ox, oy, yaw, vid, trace=TRACE and t == 0))
    n_ok = sum(r[0] for r in res)
    lift = np.mean([r[1] for r in res]) * 1000
    held = np.mean([r[2] for r in res]) * 1000
    falls = sum(r[3] for r in res)
    summary.append((key, n_ok))
    print(f"{spec.name_ja:20s} {n_ok:2d}/{TRIALS:<2d}  {lift:8.1f}      {held:8.1f}      {falls}", flush=True)
print("RESULT " + " ".join(f"{k}={n}/{TRIALS}" for k, n in summary))
