"""Run an exported walking ONNX in the GPU training sim (mjlab) with fixed commands and
report the net yaw / displacement, to compare against the CPU MuJoCo runner (sim2sim).

usage: python scripts/gpu_turn_test.py <walk.onnx> [task=Mjlab-SearchWalk-Flat-MicroDuck]
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import onnxruntime as ort
import torch

import microduck_fetch.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

ONNX = sys.argv[1]
TASK = sys.argv[2] if len(sys.argv) > 2 else "Mjlab-SearchWalk-Flat-MicroDuck"
CASES = [(0, 0.7, 0.45), (0, -0.7, 0.85), (0, 0.5, 0.45), (0.3, 0.0, 0.45), (0.3, 0.0, 0.85),
         (0.25, 0.4, 0.6), (-0.25, 0.0, 0.6)]
n = len(CASES)

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = n
for name in ("push_robot",):
    cfg.events.pop(name, None)
tw = cfg.commands["twist"]
tw.resampling_time_range = (1e6, 1e6)
tw.rel_standing_envs = 0.0
if hasattr(tw, "rel_turn_in_place_envs"):
    tw.rel_turn_in_place_envs = 0.0
cfg.commands["head_pose"].resampling_time_range = (1e6, 1e6)
cfg.episode_length_s = 30.0
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
sess = ort.InferenceSession(ONNX)
iname = sess.get_inputs()[0].name


def act(o):
    x = o.cpu().numpy().astype(np.float32)          # exported graph has batch size 1
    return np.concatenate([sess.run(None, {iname: x[i:i + 1]})[0] for i in range(len(x))], 0)

obs, _ = env.reset()
twist = env.command_manager.get_term("twist")
head = env.command_manager.get_term("head_pose")
robot = env.scene["robot"]


def set_cmds(on):
    for i, (vx, wz, hp) in enumerate(CASES):
        twist.vel_command_b[i] = torch.tensor([vx, 0.0, wz] if on else [0.0, 0.0, 0.0], device=env.device)
        if hasattr(twist, "is_standing_env"):
            twist.is_standing_env[i] = not on
        head._command[i] = torch.tensor([0.0, hp, 0.0, 0.0], device=env.device)


def yaw_of():
    q = robot.data.root_link_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).cpu().numpy()


set_cmds(False)
for _ in range(25):
    a = act(obs["actor"])
    obs, *_ = env.step(torch.as_tensor(a, device=env.device))
    set_cmds(False)
p0 = robot.data.root_link_pos_w[:, :2].cpu().numpy().copy()
prev = yaw_of(); net = np.zeros(n)
set_cmds(True)
obs = env.observation_manager.compute()
for _ in range(275):
    a = act(obs["actor"])
    obs, *_ = env.step(torch.as_tensor(a, device=env.device))
    set_cmds(True)
    y = yaw_of(); net += (y - prev + np.pi) % (2 * np.pi) - np.pi; prev = y
p1 = robot.data.root_link_pos_w[:, :2].cpu().numpy()
for i, (vx, wz, hp) in enumerate(CASES):
    d = p1[i] - p0[i]
    print(f"GPU walk={os.path.basename(ONNX)} cmd=({vx},{wz}) head={hp} net_yaw={math.degrees(net[i]):+5.0f}deg "
          f"(cmd {math.degrees(wz*5.5):+5.0f}) disp=({d[0]:+.2f},{d[1]:+.2f})", flush=True)
