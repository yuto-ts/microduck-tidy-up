"""Fetch demo: Microduck searches the floor for objects, walks up, picks each one with its
beak and drops it in a tray.

usage: python scripts/fetch_demo.py <pick.onnx> <out.mp4|-> [seed=0] [n_objects=3]
                                    [max_time_s=150] [narrow|wide] [walk.onnx]
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import math
import numpy as np
import mujoco

from microduck_fetch import sim as S, pick_profile as P
from microduck_fetch.props import CATALOG
from microduck_fetch.runner import PickRunner
from microduck_fetch.fetch import FrontCamPerception, FetchController, FetchParams, Tray

ONNX = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
N_OBJ = int(sys.argv[4]) if len(sys.argv) > 4 else 3
T_MAX = float(sys.argv[5]) if len(sys.argv) > 5 else 150.0
WIDE = len(sys.argv) > 6 and sys.argv[6] == "wide"
WALK = sys.argv[7] if len(sys.argv) > 7 else None      # walking ONNX (default: stock alpha)
POOL = ["wood_cube_20", "dice_16", "lego_2x4", "sponge", "eraser_side", "mini_car"]

rng = np.random.default_rng(SEED)
tray = Tray()
objs = []
while len(objs) < N_OBJ:
    r, a = rng.uniform(0.45, 1.0), rng.uniform(-math.pi, math.pi)
    xy = (r * math.cos(a), r * math.sin(a))
    if math.hypot(xy[0] - tray.center[0], xy[1] - tray.center[1]) < 0.40:
        continue
    if any(math.hypot(xy[0] - o[2][0], xy[1] - o[2][1]) < 0.25 for o in objs):
        continue
    key = POOL[len(objs) % len(POOL)]
    objs.append((f"obj{len(objs)}_{key}", CATALOG[key], xy, rng.uniform(-math.pi, math.pi)))

for n, _, xy, _ in objs:
    print(f"OBJ {n} at ({xy[0]:+.2f},{xy[1]:+.2f})")
sim = S.build(S.SCENE_FLAT, catalog=objs, extra=tray.add_to)
m, d = sim.model, sim.data
clip = P.WIDE_OFFSET_CLIP if WIDE else P.OFFSET_CLIP
run = PickRunner(sim, ONNX, offset_clip=clip, **({'walk_onnx': WALK} if WALK else {}))
run.reset_standing()
targets = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{n}_geom") for n, *_ in objs}
perc = FrontCamPerception(m, d, targets)
params = (FetchParams(aim_dx=0.0, aim_dy=0.0, win_x=(-P.WIDE_NOISE_X, P.WIDE_NOISE_X),
                      win_y=(-P.WIDE_NOISE_Y, P.WIDE_NOISE_Y)) if WIDE else FetchParams())
ctl = FetchController(run, perc, tray, params)

writer = None
if OUT:
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    W, H = 960, 540
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    vr = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera(); cam.distance = 0.8; cam.azimuth = 150; cam.elevation = -32
    perc.want_rgb = True
    jp_font = os.environ.get("JP_FONT", "/mnt/c/Windows/Fonts/meiryo.ttc")   # any Japanese TTF/TTC
    font = ImageFont.truetype(jp_font, 20)
    small = ImageFont.truetype(jp_font, 15)
    writer = imageio.get_writer(OUT, fps=25, codec="libx264", quality=7, macro_block_size=1)
    STATE_JA = {"SEARCH": "探索(その場で回る)", "WANDER": "探索(移動)", "APPROACH": "接近",
                "FINAL": "最終接近", "SETTLE": "位置確認", "STEP": "微調整(前へ)", "BACKUP": "後退して再接近",
                "PICKING": "拾う(BeakPick)", "CARRY": "片付けエリアへ運ぶ", "DROP": "くちばしを開いて置く",
                "LEAVE": "離れる"}

look = np.array([0.0, 0.0, 0.1])
steps = int(T_MAX * 50)
for k in range(steps):
    ctl.step()
    if os.environ.get("TRACE") == "1" and k % 100 == 0:
        x0, y0, yaw = run.root_xy_yaw()
        print(f"  t={k*0.02:5.1f} st={ctl.state:8s} pos=({x0:+.2f},{y0:+.2f}) yaw={math.degrees(yaw):+5.0f} "
              f"turned={math.degrees(ctl.turned):4.0f} trunk_z={run.trunk_z()*1000:.0f} "
              f"pick={run.picking} vel={np.round(run.pol.vel_cmd,2)} "
              f"dets={[det.name.split('_')[0] for det in ctl.detections]} mem={ {n.split('_')[0]: np.round(v,2).tolist() for n,v in ctl.memory.items()} }", flush=True)
    if len(ctl.delivered) == N_OBJ and ctl.state == "SEARCH":
        break
    if writer is not None and k % 2 == 0:
        x0, y0, _ = run.root_xy_yaw()
        look = 0.95 * look + 0.05 * np.array([x0, y0, 0.1])
        cam.lookat[:] = look
        vr.update_scene(d, camera=cam)
        vr.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False      # shadow-map streaks
        vr.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        im = Image.fromarray(vr.render())
        dr = ImageDraw.Draw(im, "RGBA")
        state = "拾う(BeakPick)" if run.picking else STATE_JA.get(ctl.state, ctl.state)
        dr.rectangle([0, 0, W, 34], fill=(0, 0, 0, 150))
        dr.text((10, 5), f"{k*0.02:5.1f}s  状態: {state}   片付け済み: {len(ctl.delivered)}/{N_OBJ}"
                + (f"   目標: {ctl.target.split('_', 1)[1]}" if ctl.target else ""),
                font=font, fill=(255, 255, 255))
        if perc.last_rgb is not None:
            sc = 2
            inset = Image.fromarray(perc.last_rgb).resize((perc.W * sc, perc.H * sc), Image.NEAREST)
            di = ImageDraw.Draw(inset)
            for det in ctl.detections:
                u0, v0, u1, v1 = det.bbox
                di.rectangle([u0 * sc - 2, v0 * sc - 2, u1 * sc + 2, v1 * sc + 2],
                             outline=(255, 60, 60), width=2)
                di.text((u0 * sc, max(0, v0 * sc - 16)), det.name.split("_", 1)[1], font=small,
                        fill=(255, 255, 0))
            im.paste(inset, (W - perc.W * sc - 10, H - perc.H * sc - 10))
            dr.text((W - perc.W * sc - 10, H - perc.H * sc - 32), "頭のカメラ(検出)", font=small,
                    fill=(255, 255, 255))
        writer.append_data(np.asarray(im))
if writer is not None:
    for _ in range(25):
        writer.append_data(np.asarray(im))
    writer.close()

t = (k + 1) * 0.02
print(f"RESULT seed={SEED} delivered={len(ctl.delivered)}/{N_OBJ} time={t:.1f}s "
      f"failed_picks={sum(ctl.failed.values())} policy={os.path.basename(ONNX)} window={'wide' if WIDE else 'narrow'}")
for e in ctl.log:
    print("  ", e)
