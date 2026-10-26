"""Run the stock ground-pick policy on the beak model and log where the beak goes.

usage: python scripts/pick_probe.py [cube_x cube_y] [open_at_phase close_at_phase] [video.mp4|-] [mass_g]
  Without a cube: just logs the mouth-tip trajectory. With a cube at (cube_x, cube_y)
  (relative to the robot's start, metres): opens the beak at `open_at_phase`, shuts it at
  `close_at_phase`, and reports whether the cube came up with the head.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, mujoco
from microduck_fetch import sim as S

from microduck_fetch.runner import POLICIES as POL
args = sys.argv[1:]
cube_xy = (float(args[0]), float(args[1])) if len(args) >= 2 else None
ph_open = float(args[2]) if len(args) > 2 else 0.05
ph_close = float(args[3]) if len(args) > 3 else 0.45
OUT = args[4] if len(args) > 4 and args[4] != "-" else None
MASS = float(args[5]) / 1000 if len(args) > 5 else 0.010
H = float(os.environ.get('PROP_H', '0.020')); SIZE = (0.020, 0.020, H)
ip = S.upstream_infer()
props = [S.Prop("cube", (cube_xy[0], cube_xy[1], H / 2), size=SIZE, mass=MASS)] if cube_xy else []
sim = S.build(S.SCENE_FLAT, props); m, d = sim.model, sim.data
pol = ip.PolicyInference(m, d, bam_ctrl=sim.bam, standing_onnx_path=f"{POL}/alpha_stand.onnx",
                         walking_onnx_path=f"{POL}/alpha_walking.onnx",
                         ground_pick_onnx_path=f"{POL}/alpha_ground_pick.onnx", ground_pick_period=4.0,
                         new_cmd_obs=True, use_projected_gravity=True)
d.qpos[0:3] = [0, 0, 0.125]; d.qpos[3:7] = [1, 0, 0, 0]
for i, qi in enumerate(pol.joint_qpos_indices): d.qpos[qi] = pol.default_pose[i]
sim.bam.reset(d.qpos); pol.set_position_targets(pol.default_pose); mujoco.mj_forward(m, d)
tip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
ca = sim.props.get("cube")
writer = None
if OUT:
    import imageio
    r = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera()
    cam.distance = 0.45; cam.azimuth = 60; cam.elevation = -15
    writer = imageio.get_writer(OUT, fps=25, codec="libx264", macro_block_size=1)
T_TRIG = 1.0; low = (1.0, None)
for k in range(int(6.0 * 50)):
    t = k / 50
    if abs(t - T_TRIG) < 1e-6: pol.trigger_ground_pick()
    ph = pol.ground_pick_phase if pol.ground_pick_mode else (None if t < T_TRIG else 1.0)
    if ph is not None and cube_xy:
        OPEN_FRAC = float(os.environ.get('OPEN_FRAC', '1.0'))
        sim.beak.set_open(OPEN_FRAC if ph_open <= ph < ph_close else 0.0)
    pol.update_ground_pick_phase(0.02); pol.update_behavior(0.02)
    pol.apply_action(pol.infer()); sim.step(4)
    tz = d.site_xpos[tip][2]
    if tz < low[0]: low = (tz, t, d.site_xpos[tip].copy(), ph)
    if writer and k % 2 == 0:
        cam.lookat[:] = [d.qpos[0] + 0.05, d.qpos[1], 0.07]; r.update_scene(d, camera=cam); writer.append_data(r.render())
    if k % 10 == 0 and t >= T_TRIG - 0.01:
        z_axis = d.site_xmat[tip].reshape(3, 3)[:, 2]
        cs = "" if ca is None else f" cube=({d.qpos[ca]*1000:5.0f},{d.qpos[ca+1]*1000:4.0f},{d.qpos[ca+2]*1000:4.0f})mm"
        print(f"t={t:4.2f} phase={'-' if ph is None else f'{ph:.2f}'} policy={pol.current_policy:11s} "
              f"tip=({d.site_xpos[tip][0]*1000:5.1f},{d.site_xpos[tip][1]*1000:5.1f},{tz*1000:5.1f})mm "
              f"tip_z_axis={np.round(z_axis,2)} mouth={sim.beak.angle:.2f}{cs} trunk_z={d.qpos[2]*1000:.0f}")
if writer: writer.close()
print(f"LOWEST tip z={low[0]*1000:.1f}mm at t={low[1]:.2f}s phase={low[3]} xyz={np.round(low[2]*1000,1)}")
if ca is not None:
    up = d.qpos[ca + 2] > H / 2 + 0.02
    print(f"RESULT H={H*1000:.0f}mm open_frac={os.environ.get('OPEN_FRAC','1.0')} cube=({cube_xy[0]},{cube_xy[1]}) open@{ph_open} close@{ph_close} final cube z={d.qpos[ca+2]*1000:.0f}mm "
          f"-> {'PICKED' if up else 'NOT PICKED'}  trunk_z={d.qpos[2]*1000:.0f}mm")
