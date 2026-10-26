"""Render the candidate objects laid out in front of Microduck (for the report / video).

usage: python scripts/showcase.py [out.png] [flat|apartment]
Graspable candidates go in an arc in front of the robot; borderline ones (too low / too
wide / too heavy) in a second row, tinted by their own colours.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import mujoco
import imageio

from microduck_fetch import sim as S
from microduck_fetch.props import GRASPABLE, BORDERLINE

OUT = sys.argv[1] if len(sys.argv) > 1 else "showcase.png"
scene = S.SCENE_APARTMENT if len(sys.argv) > 2 and sys.argv[2] == "apartment" else S.SCENE_FLAT
cat = []
for i, p in enumerate(GRASPABLE):           # front arc
    a = np.radians(-60 + 120 * i / (len(GRASPABLE) - 1))
    cat.append((f"g_{p.key}", p, (0.20 * np.cos(a), 0.20 * np.sin(a)), 0.0))
for i, p in enumerate(BORDERLINE):          # outer row
    a = np.radians(-30 + 60 * i / (len(BORDERLINE) - 1))
    cat.append((f"b_{p.key}", p, (0.30 * np.cos(a), 0.30 * np.sin(a)), 0.0))
sim = S.build(scene, catalog=cat, bam=False)
m, d = sim.model, sim.data
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND"))
mujoco.mj_forward(m, d)
m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
r = mujoco.Renderer(m, 720, 1280)
views = []
for az, el, dist, look in [(180, -35, 0.75, (0.15, 0, 0.03)), (210, -20, 0.55, (0.12, 0.05, 0.05))]:
    cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = az, el, dist
    cam.lookat[:] = look
    r.update_scene(d, camera=cam); views.append(r.render())
imageio.imwrite(OUT, np.concatenate(views, 0))
print("saved", OUT)
for name, p, xy, _ in cat:
    print(f"{name:22s} {p.name_ja:20s} size={tuple(round(s*1000,1) for s in p.size)}mm "
          f"mass={p.mass*1000:.1f}g  at ({xy[0]*1000:.0f},{xy[1]*1000:.0f})mm  {p.note}")
