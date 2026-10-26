"""MDP terms for the BeakPick task (reward / observation / event functions).

Phase helpers come from upstream (`mjlab_microduck.tasks.mdp`): the ground-pick phase
command encodes the phase as [cos 2*pi*phi, sin 2*pi*phi] in the twist slot.
"""
from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks.mdp import _gp_phase, phase_pose_blend, phase_rise_gate

_GRASP = SceneEntityCfg("robot", site_names=["beak_grasp"])
_MOUTH = SceneEntityCfg("robot", joint_names=["passive_mouth"])


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def _yaw_cs(q: torch.Tensor):
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return torch.cos(yaw), torch.sin(yaw)


def _prop_in_yaw_frame(env, prop_name: str):
    robot = env.scene["robot"]
    rel = env.scene[prop_name].data.root_link_pos_w - robot.data.root_link_pos_w
    c, s = _yaw_cs(robot.data.root_link_quat_w)
    x = c * rel[:, 0] + s * rel[:, 1]
    y = -s * rel[:, 0] + c * rel[:, 1]
    return x, y


def _grasp_dist(env, prop_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    site = robot.data.site_pos_w[:, asset_cfg.site_ids[0], :]
    d = torch.linalg.norm(env.scene[prop_name].data.root_link_pos_w - site, dim=-1)
    return torch.nan_to_num(d, nan=1.0)


def _prop_height(env, prop_name: str) -> torch.Tensor:
    z = env.scene[prop_name].data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
    return torch.nan_to_num(z, nan=0.0)


# -- per-step hook (registered as a zero-weight reward, the upstream pattern) ------------
def beak_phase_control(env, open_start: float, close_phase: float, open_angle: float,
                       command_name: str = "twist", asset_cfg: SceneEntityCfg = _MOUTH):
    """Drive the mouth position target from the phase: open in [open_start, close_phase).

    Scripted, the way the real robot's daemon drives the mouth outside the policy.
    Returns zeros (not a real reward).
    """
    robot = env.scene[asset_cfg.name]
    ph = _gp_phase(env, command_name)
    is_open = (ph >= open_start) & (ph < close_phase)
    target = torch.where(is_open, torch.full_like(ph, open_angle), torch.zeros_like(ph))
    robot.set_joint_position_target(target.unsqueeze(1), joint_ids=asset_cfg.joint_ids)
    return torch.zeros(env.num_envs, device=env.device)


# -- rewards ----------------------------------------------------------------------------
def grasp_approach(env, std: float, prop_name: str = "prop", command_name: str = "twist",
                   descent_end: float = 0.3, hold_end: float = 0.45, rise_end: float = 0.8,
                   asset_cfg: SceneEntityCfg = _GRASP):
    """Down phase: bring the beak's grasp point onto the prop centre (Gaussian kernel)."""
    d = _grasp_dist(env, prop_name, asset_cfg)
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * torch.exp(-(d / std) ** 2)


def prop_lifted(env, lift_height: float, hold_std: float, prop_name: str = "prop",
                rest_z: float = 0.010, command_name: str = "twist",
                hold_end: float = 0.45, rise_end: float = 0.8,
                asset_cfg: SceneEntityCfg = _GRASP):
    """Up phase: prop height above its resting height (saturating), counted only while the
    prop is still at the grasp point, so flicking it into the air earns nothing."""
    h = ((_prop_height(env, prop_name) - rest_z) / lift_height).clamp(0.0, 1.0)
    held = torch.exp(-(_grasp_dist(env, prop_name, asset_cfg) / hold_std) ** 2)
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * held * h


def prop_held(env, std: float, prop_name: str = "prop", command_name: str = "twist",
              hold_end: float = 0.45, rise_end: float = 0.8,
              asset_cfg: SceneEntityCfg = _GRASP):
    """Up phase: keep the prop at the grasp point (dense companion to prop_lifted)."""
    d = _grasp_dist(env, prop_name, asset_cfg)
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * torch.exp(-(d / std) ** 2)


def prop_knocked(env, prop_name: str = "prop", close_phase: float = 0.36,
                 command_name: str = "twist"):
    """Before the beak shuts: horizontal prop speed (pushing or toppling it). Cost >= 0."""
    v = env.scene[prop_name].data.root_link_lin_vel_w[:, :2]
    gate = (_gp_phase(env, command_name) < close_phase).to(v.dtype)
    return gate * torch.nan_to_num(torch.linalg.norm(v, dim=-1), nan=0.0).clamp(max=2.0)


def pick_success(env, lift_height: float = 0.04, hold_std: float = 0.02,
                 prop_name: str = "prop", rest_z: float = 0.010, phase_lo: float = 0.8,
                 command_name: str = "twist", asset_cfg: SceneEntityCfg = _GRASP):
    """Metric (weight 0, logged only): 1 while back up with the prop in the beak."""
    ok = ((_prop_height(env, prop_name) - rest_z) > lift_height) & \
         (_grasp_dist(env, prop_name, asset_cfg) < hold_std)
    return (ok & (_gp_phase(env, command_name) >= phase_lo)).float()


# -- observations -----------------------------------------------------------------------
def prop_offset_command(env, nominal: tuple, prop_name: str = "prop", noise: float = 0.0,
                        clip: float = 0.05):
    """6-D body_command slot: [dx, dy, 0, 0, 0, 0], the prop position relative to the
    nominal pick spot in the robot's yaw frame. Reuses the zero-padded body_pose slot so
    the policy keeps the shared 61-D observation layout. At deployment the camera (or the
    navigation layer) fills these two numbers."""
    x, y = _prop_in_yaw_frame(env, prop_name)
    dx, dy = x - nominal[0], y - nominal[1]
    if noise > 0:
        dx = dx + (torch.rand_like(dx) * 2 - 1) * noise
        dy = dy + (torch.rand_like(dy) * 2 - 1) * noise
    out = torch.zeros(env.num_envs, 6, device=env.device)
    out[:, 0] = torch.nan_to_num(dx, nan=0.0).clamp(-clip, clip)
    out[:, 1] = torch.nan_to_num(dy, nan=0.0).clamp(-clip, clip)
    return out


def prop_state_privileged(env, prop_name: str = "prop"):
    """Critic only: prop position + linear velocity in the robot base frame (6-D)."""
    robot = env.scene["robot"]
    prop = env.scene[prop_name]
    rot_t = _quat_to_mat(robot.data.root_link_quat_w).transpose(1, 2)
    rel = prop.data.root_link_pos_w - robot.data.root_link_pos_w
    p = torch.bmm(rot_t, rel.unsqueeze(-1)).squeeze(-1)
    v = torch.bmm(rot_t, prop.data.root_link_lin_vel_w.unsqueeze(-1)).squeeze(-1)
    return torch.nan_to_num(torch.cat([p, v], dim=-1), nan=0.0).clamp(-5.0, 5.0)


def mouth_angle(env, asset_cfg: SceneEntityCfg = _MOUTH):
    """Critic only: beak opening (rad)."""
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


# -- events -----------------------------------------------------------------------------
def reset_prop_in_front(env, env_ids, nominal: tuple, noise_x: float, noise_y: float,
                        rest_z: float, yaw_range: float = 0.6, prop_name: str = "prop"):
    """Place the prop at the nominal pick spot (+uniform noise, random yaw) in front of the
    robot. Reads the robot root from qpos (final pose after reset_base): register AFTER
    the robot reset events (events run in insertion order)."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    robot = env.scene["robot"]
    prop = env.scene[prop_name]
    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    c, s = _yaw_cs(root[:, 3:7])
    n = len(env_ids)
    ox = nominal[0] + (torch.rand(n, device=env.device) * 2 - 1) * noise_x
    oy = nominal[1] + (torch.rand(n, device=env.device) * 2 - 1) * noise_y
    yaw = torch.atan2(s, c) + (torch.rand(n, device=env.device) * 2 - 1) * yaw_range
    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = root[:, 0] + c * ox - s * oy
    pose[:, 1] = root[:, 1] + s * ox + c * oy
    pose[:, 2] = env.scene.terrain.env_origins[env_ids, 2] + rest_z + 1e-4
    pose[:, 3] = torch.cos(yaw / 2)
    pose[:, 6] = torch.sin(yaw / 2)
    prop.write_root_link_pose_to_sim(pose, env_ids)
    prop.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids)
