"""Run one pick per candidate object and tile the clips into a single labelled video.

Each tile: the object's name, its size/mass, and the outcome (成功 / 失敗) shown after the
pick. Placement is the nominal spot + noise drawn from `seed` (same for every object).

usage: python scripts/make_montage.py <pick.onnx> <out.mp4> [keys=all|k1,k2] [seed=0] [cols=4]

Each object is simulated in its own subprocess (`--clip`), cached as <out>_clips/<key>.mp4
plus a result line: rendering many EGL contexts in one process crashes after ~12.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont

from microduck_fetch import sim as S, pick_profile as P
from microduck_fetch.props import CATALOG, GRASPABLE, BORDERLINE
from microduck_fetch.runner import PickRunner

CLIP_MODE = sys.argv[1] == "--clip"
if CLIP_MODE:
    sys.argv.pop(1)
ONNX, OUT = sys.argv[1], sys.argv[2]
KEYS = ([p.key for p in GRASPABLE + BORDERLINE] if len(sys.argv) <= 3 or sys.argv[3] == "all"
        else sys.argv[3].split(","))
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
COLS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
W, H = 400, 300
FONT = os.environ.get("JP_FONT", "/mnt/c/Windows/Fonts/meiryo.ttc")   # any Japanese TTF/TTC
font = ImageFont.truetype(FONT, 18)
font_big = ImageFont.truetype(FONT, 30)
font_title = ImageFont.truetype(FONT, 26)
BORDER_KEYS = {p.key for p in BORDERLINE}


def run_one(spec, ox, oy, yaw):
    sim = S.build(S.SCENE_FLAT, catalog=[("target", spec, (P.NOMINAL[0] + ox, P.NOMINAL[1] + oy), yaw)])
    sim.model.vis.global_.offwidth, sim.model.vis.global_.offheight = W, H
    run = PickRunner(sim, ONNX)
    run.reset_standing()
    rest_z = run.prop_pos("target")[2]
    r = mujoco.Renderer(sim.model, H, W)
    cam = mujoco.MjvCamera(); cam.distance = 0.58; cam.azimuth = 135; cam.elevation = -14
    frames = []
    for k in range(int(7.0 * 50)):
        if k == 50:
            run.start_pick("target")
        run.step()
        if k % 2 == 0:
            x0, y0, _ = run.root_xy_yaw()
            cam.lookat[:] = [x0 + 0.04, y0, 0.12]   # whole robot incl. the raised head
            r.update_scene(sim.data, camera=cam)
            frames.append(r.render().copy())
    lift = run.prop_pos("target")[2] - rest_z
    ok = lift > 0.03 and run.grasp_dist("target") < 0.03 and run.trunk_z() > 0.09
    r.close()
    return frames, ok


def label(frame, spec, ok, show_result, phase_txt):
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, W, 52], fill=(0, 0, 0, 150))
    tag = "  [限界例]" if spec.key in BORDER_KEYS else ""
    d.text((8, 3), spec.name_ja + tag, font=font, fill=(255, 255, 255))
    if spec.shape == "sphere":
        dims = f"直径{spec.size[0]*1000:.0f}mm"
    elif spec.shape == "box":
        dims = "×".join(f"{s*1000:.0f}" for s in spec.size) + "mm"
    else:
        dims = f"直径{spec.size[0]*1000:.1f}×高さ{spec.size[1]*1000:.0f}mm"
    d.text((8, 27), f"{dims}  {spec.mass*1000:.0f}g", font=font, fill=(210, 210, 210))
    d.text((W - 8, H - 26), phase_txt, font=font, fill=(255, 255, 255), anchor="ra")
    if show_result:
        col = (40, 200, 90, 220) if ok else (220, 60, 60, 220)
        txt = "成功" if ok else "失敗"
        d.rounded_rectangle([W - 96, 60, W - 8, 104], 8, fill=col)
        d.text((W - 52, 82), txt, font=font_big, fill=(255, 255, 255), anchor="mm")
    return np.asarray(im)


rng = np.random.default_rng(SEED)
ox, oy, yaw = rng.uniform(-P.NOISE_X, P.NOISE_X), rng.uniform(-P.NOISE_Y, P.NOISE_Y), rng.uniform(-0.6, 0.6)
CLIP_DIR = os.path.splitext(OUT)[0] + "_clips"
os.makedirs(CLIP_DIR, exist_ok=True)

if CLIP_MODE:                       # child: one object -> <key>.mp4 + <key>.txt
    key = KEYS[0]
    frames, ok = run_one(CATALOG[key], ox, oy, yaw)
    imageio.mimwrite(os.path.join(CLIP_DIR, f"{key}.mp4"), frames, fps=25, codec="libx264",
                     quality=9, macro_block_size=1)
    open(os.path.join(CLIP_DIR, f"{key}.txt"), "w").write("1" if ok else "0")
    raise SystemExit(0)

import subprocess
clips = []
for key in KEYS:
    spec = CATALOG[key]
    if not os.path.exists(os.path.join(CLIP_DIR, f"{key}.txt")):   # resume: reuse finished clips
        subprocess.run([sys.executable, __file__, "--clip", ONNX, OUT, key, str(SEED), str(COLS)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = open(os.path.join(CLIP_DIR, f"{key}.txt")).read().strip() == "1"
    frames = [f for f in imageio.get_reader(os.path.join(CLIP_DIR, f"{key}.mp4"))]
    print(f"{spec.name_ja:20s} {'成功' if ok else '失敗'}", flush=True)
    clips.append((spec, frames, ok))

n = len(clips)
rows = (n + COLS - 1) // COLS
T = min(len(c[1]) for c in clips)
HOLD = 25                                        # freeze the last frame 1 s at the end
TITLE_H = 44
n_ok = sum(c[2] for c in clips)
title = (f"Microduck BeakPick  ({os.path.basename(ONNX)})   成功 {n_ok}/{n}   "
         "等速再生・CPU MuJoCo + BAM サーボモデル")
writer = imageio.get_writer(OUT, fps=25, codec="libx264", quality=8, macro_block_size=1)
for t in range(T + HOLD):
    tt = min(t, T - 1)
    sim_t = tt * 2 / 50                          # seconds (frames every 2 control steps)
    if sim_t < 1.0:
        ph = "立って待機"
    elif sim_t < 1.0 + P.GP_PERIOD * P.DESCENT_END:
        ph = "しゃがむ・口を開く"
    elif sim_t < 1.0 + P.GP_PERIOD * P.HOLD_END:
        ph = "挟む"
    elif sim_t < 1.0 + P.GP_PERIOD * P.RISE_END:
        ph = "持ち上げる"
    else:
        ph = "保持"
    canvas = np.zeros((TITLE_H + rows * H, COLS * W, 3), np.uint8) + 30
    for i, (spec, frames, ok) in enumerate(clips):
        r_, c_ = divmod(i, COLS)
        tile = label(frames[tt], spec, ok, show_result=sim_t >= 1.0 + P.GP_PERIOD * P.RISE_END,
                     phase_txt=f"{sim_t:4.1f}s {ph}")
        canvas[TITLE_H + r_ * H:TITLE_H + (r_ + 1) * H, c_ * W:(c_ + 1) * W] = tile
    im = Image.fromarray(canvas)
    ImageDraw.Draw(im).text((12, 6), title, font=font_title, fill=(255, 255, 255))
    writer.append_data(np.asarray(im))
writer.close()
print(f"saved {OUT}  ({n_ok}/{n} 成功, {(T + HOLD) / 25:.1f}s)")
