"""Map pick success over the object's offset (dx, dy) from the nominal pick spot.

usage: python scripts/pick_window_map.py <pick.onnx> [object_key=wood_cube_20] [clip=0.05]
Prints a grid: rows dy, columns dx (mm). One trial per cell, random yaw (fixed seed).
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np

from microduck_tidy import sim as S, pick_profile as P
from microduck_tidy.props import CATALOG
from microduck_tidy.runner import PickRunner

ONNX = sys.argv[1]
KEY = sys.argv[2] if len(sys.argv) > 2 else "wood_cube_20"
CLIP = float(sys.argv[3]) if len(sys.argv) > 3 else P.OFFSET_CLIP
DX = [-30, -15, 0, 15, 30, 45]
DY = [-40, -20, 0, 20, 40]
rng = np.random.default_rng(1)
grid = {}
for dy in DY:
    for dx in DX:
        xy = (P.NOMINAL[0] + dx / 1000, P.NOMINAL[1] + dy / 1000)
        sim = S.build(S.SCENE_FLAT, catalog=[("target", CATALOG[KEY], xy, rng.uniform(-0.6, 0.6))])
        run = PickRunner(sim, ONNX, offset_clip=CLIP)
        run.reset_standing()
        rest = run.prop_pos("target")[2]
        for k in range(int(7.0 * 50)):
            if k == 50:
                run.start_pick("target")
            run.step()
        ok = (run.prop_pos("target")[2] - rest > 0.03 and run.grasp_dist("target") < 0.03
              and run.trunk_z() > 0.09)
        grid[(dx, dy)] = ok
        print(f"cell dx={dx:+d} dy={dy:+d} -> {'OK' if ok else '--'}", flush=True)
print(f"MAP {os.path.basename(ONNX)} {KEY} (rows dy mm, cols dx mm)")
print("      " + "".join(f"{dx:+5d}" for dx in DX))
for dy in DY:
    print(f"{dy:+5d} " + "".join("   OK" if grid[(dx, dy)] else "   --" for dx in DX))
print(f"MAP_TOTAL {sum(grid.values())}/{len(grid)}")
