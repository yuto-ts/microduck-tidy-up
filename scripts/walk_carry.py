"""Does the stock walking policy still walk with the beak model, and while carrying a prop?

usage: python scripts/walk_carry.py <mode> [vx] [secs] [walk.onnx|-] [vy] [wz] [flat|apartment]
  mode: upstream | beak | carry10 | carry20 | carry30   (carryNN = NN-gram 20 mm cube held in the beak)
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, mujoco
from microduck_fetch import sim as S

from microduck_fetch.runner import POLICIES as POL
mode = sys.argv[1]
vx = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
WALK = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else os.path.join(POL, "alpha_walking.onnx")
STAND = os.path.join(POL, "alpha_stand.onnx")
vy = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
wz = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
SCENE = S.SCENE_APARTMENT if (len(sys.argv) > 7 and sys.argv[7] == "apartment") else S.SCENE_FLAT
ip = S.upstream_infer()

if mode == "upstream":
    bm = ip.load_bam_model(ip.BAM_KP_FW, 7.4, None)
    model, data, bam, _ = ip.load_mujoco_with_bam(SCENE, bm, 0.005, 0.1, ip.BAM_VIN_MIN)
    beak = None; cube = None
else:
    props = [S.Prop("cube", (-3.5, 2.5, 0.01), mass=int(mode[5:]) / 1000)] if mode.startswith("carry") else []
    sim = S.build(SCENE, props)
    model, data, bam, beak = sim.model, sim.data, sim.bam, sim.beak
    cube = sim.props.get("cube")

policy = ip.PolicyInference(model, data, bam_ctrl=bam, walking_onnx_path=WALK,
                            standing_onnx_path=STAND, new_cmd_obs=True, use_projected_gravity=True)
fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"); qa = model.jnt_qposadr[fj]
data.qpos[qa:qa + 3] = [0, 0, 0.125]; data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
for i, qi in enumerate(policy.joint_qpos_indices):
    data.qpos[qi] = policy.default_pose[i]
if cube is not None:
    S.place_in_beak(sim, "cube")
bam.reset(data.qpos); policy.set_position_targets(policy.default_pose); mujoco.mj_forward(model, data)
tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")

zs = []; held = True
for k in range(int(secs * 50)):
    policy.set_vel_cmd(*((vx, vy, wz) if k >= 50 else (0.0, 0.0, 0.0)))      # stand 1 s, then walk
    policy.update_ground_pick_phase(0.02); policy.update_behavior(0.02)
    a = policy.infer(); policy.apply_action(a)
    for _ in range(4):
        bam.update()
        if beak is not None: beak.update()
        mujoco.mj_step(model, data)
    if cube is not None and k < 10:
        S.hold_in_gap(sim, "cube")       # hand-over: keep the cube between the pads while the beak shuts
    zs.append(data.qpos[qa + 2])
    if cube is not None and k >= 10 and np.linalg.norm(data.qpos[cube:cube + 3] - data.site_xpos[tip]) > 0.05:
        held = False
fell = min(zs[50:]) < 0.08
cube_s = "" if cube is None else f" cube={'HELD' if held else 'DROPPED'} mouth={beak.angle:.2f}rad"
print(f"RESULT {os.path.basename(SCENE)[6:-4]:9s} mode={mode:8s} cmd=({vx},{vy},{wz}) dist_x={data.qpos[qa]:+.3f}m "
      f"y={data.qpos[qa+1]:+.3f} min_z={min(zs[50:])*1000:.0f}mm {'FELL' if fell else 'upright'}{cube_s}")
