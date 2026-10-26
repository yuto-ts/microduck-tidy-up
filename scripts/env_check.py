"""Sanity-check the BeakPick training env on the GPU sim (no learning).

Prints: action dim, which geoms collide, beak angle vs phase under a zero action (robot
holds HOME), prop placement relative to the robot, and the grasp-point distance.
usage: python scripts/env_check.py [num_envs=4] [steps=260]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch
import mujoco

import microduck_tidy.tasks  # noqa: F401
from microduck_tidy.tasks.beak_pick_env_cfg import make_beak_pick_env_cfg, NOMINAL
from microduck_tidy.tasks import mdp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 260
UPSTREAM = os.environ.get("UPSTREAM_GP") == "1"
if UPSTREAM:
    from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import make_microduck_ground_pick_env_cfg
    cfg = make_microduck_ground_pick_env_cfg()
else:
    cfg = make_beak_pick_env_cfg()
cfg.scene.num_envs = n
cfg.events.pop("push_robot", None)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
m = env.sim.mj_model
print("action dim:", env.action_manager.total_action_dim, " nu:", m.nu)
print("actuators:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])
coll = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"#{g}", int(m.geom_contype[g]), int(m.geom_conaffinity[g]),
         int(m.geom_condim[g]), np.round(m.geom_friction[g], 3).tolist())
        for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
print("colliding geoms:")
for c in coll:
    print("   ", c)

obs, _ = env.reset()
if UPSTREAM:
    act = torch.zeros(n, env.action_manager.total_action_dim, device=env.device)
    r = env.scene["robot"]
    for k in range(steps):
        env.step(act)
        if k % 10 == 0:
            print(f"k={k:3d} upstream trunk_z={r.data.root_link_pos_w[0, 2]*1000:.0f}mm")
    raise SystemExit
robot, prop = env.scene["robot"], env.scene["prop"]
grasp = SceneEntityCfg("robot", site_names=["beak_grasp"]); grasp.resolve(env.scene)
mouth = SceneEntityCfg("robot", joint_names=["passive_mouth"]); mouth.resolve(env.scene)
cmd = env.command_manager.get_command("twist")
act = torch.zeros(n, env.action_manager.total_action_dim, device=env.device)
for k in range(steps):
    obs, rew, term, trunc, extras = env.step(act)
    if k % 10 == 0:
        ph = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
        x, y = mdp._prop_in_yaw_frame(env, "prop")
        d = torch.linalg.norm(prop.data.root_link_pos_w - robot.data.site_pos_w[:, grasp.site_ids[0]], dim=-1)
        pz = prop.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
        bc = obs["actor"][0, -6:-4] if isinstance(obs, dict) else None
        print(f"k={k:3d} phase={ph[0]:.2f} mouth={robot.data.joint_pos[0, mouth.joint_ids[0]]:+.3f} "
              f"trunk_z={robot.data.root_link_pos_w[0, 2]*1000:.0f}mm prop_rel=({x[0]*1000:.0f},{y[0]*1000:.0f})mm "
              f"prop_z={pz[0]*1000:.1f}mm grasp_d={d[0]*1000:.0f}mm body_cmd_obs={None if bc is None else np.round(bc.cpu().numpy()*1000,1)}")
