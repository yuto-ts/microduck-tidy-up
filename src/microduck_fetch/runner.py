"""CPU-side runner: stock walking/standing policies + the BeakPick policy + scripted beak.

Mirrors how the real robot would run it: the policies only ever see the shared 61-D
observation; the pick policy gets the phase in the twist slot and the target's offset in
the body_pose slot; the mouth is driven outside the policy (BeakServo).
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from . import pick_profile as P
from .sim import Sim, upstream_infer, UPSTREAM

# Stock walking/standing ONNX policies (the official set: pollen-robotics/microduck-policies).
POLICIES = os.path.abspath(os.environ.get("MICRODUCK_POLICIES",
                                          os.path.join(UPSTREAM, "..", "policies")))


class PickRunner:
    """delay_lag: policy-step (20 ms) delay between the action and the servo target. The
    training actuator delays commands by 3-6 physics steps (15-30 ms); with 0 here the
    fine-tuned walking policies veer badly (sw2: -117 deg in 5.5 s vs -19 deg at lag 1)."""

    def __init__(self, sim: Sim, pick_onnx: str,
                 walk_onnx: str = os.path.join(POLICIES, "alpha_walking.onnx"),
                 stand_onnx: str = os.path.join(POLICIES, "alpha_stand.onnx"),
                 offset_clip: float = P.OFFSET_CLIP, delay_lag: int = 1):
        import onnxruntime as ort
        ip = upstream_infer()
        self.sim, self.m, self.d = sim, sim.model, sim.data
        self.offset_clip = offset_clip
        self.pol = ip.PolicyInference(self.m, self.d, bam_ctrl=sim.bam,
                                      walking_onnx_path=walk_onnx, standing_onnx_path=stand_onnx,
                                      new_cmd_obs=True, use_projected_gravity=True,
                                      delay_min_lag=delay_lag, delay_max_lag=delay_lag)
        self.pick_sess = ort.InferenceSession(pick_onnx)
        self.phase = None          # None = not picking
        self.target = None
        self.target_est = None
        fj = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.qa = int(self.m.jnt_qposadr[fj])
        self.grasp_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "beak_grasp")

    # -- setup -------------------------------------------------------------------------
    def reset_standing(self, xy=(0.0, 0.0), yaw=0.0):
        d, pol = self.d, self.pol
        d.qpos[self.qa:self.qa + 3] = [xy[0], xy[1], 0.125]
        d.qpos[self.qa + 3:self.qa + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        for i, qi in enumerate(pol.joint_qpos_indices):
            d.qpos[qi] = pol.default_pose[i]
        self.sim.bam.reset(d.qpos)
        pol.set_position_targets(pol.default_pose)
        mujoco.mj_forward(self.m, d)

    # -- geometry helpers --------------------------------------------------------------
    def root_xy_yaw(self):
        q = self.d.qpos[self.qa:self.qa + 7]
        qw, qx, qy, qz = q[3:7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return q[0], q[1], yaw

    def prop_pos(self, name):
        a = self.sim.props[name]
        return self.d.qpos[a:a + 3].copy()

    def offset_of(self, name):
        """Prop offset from the nominal pick spot, robot yaw frame (what the pick policy sees)."""
        if self.target_est is not None and name == self.target:
            return self.offset_of_xy(self.target_est)
        return self.offset_of_xy(self.prop_pos(name)[:2])

    def offset_of_xy(self, p):
        """Offset of a world point (x, y) from the nominal pick spot, robot yaw frame."""
        x0, y0, yaw = self.root_xy_yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        rx, ry = p[0] - x0, p[1] - y0
        return np.array([c * rx + s * ry - P.NOMINAL[0], -s * rx + c * ry - P.NOMINAL[1]])

    def grasp_dist(self, name):
        return float(np.linalg.norm(self.prop_pos(name) - self.d.site_xpos[self.grasp_site]))

    # -- behaviour -----------------------------------------------------------------------
    def start_pick(self, name, est_xy=None):
        """Start one pick cycle on `name`. With `est_xy` (perceived world position, frozen
        at the start) the policy's offset input comes from perception, not ground truth."""
        self.phase, self.target = 0.0, name
        self.target_est = None if est_xy is None else np.array(est_xy[:2], dtype=float)

    @property
    def picking(self):
        return self.phase is not None

    def step(self, vel_cmd=(0.0, 0.0, 0.0)):
        """One 20 ms control step (4 physics substeps)."""
        pol = self.pol
        if self.picking:
            ph = self.phase
            cmd = np.zeros(13, dtype=np.float32)
            cmd[0], cmd[1] = np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph)
            cmd[P.CMD_BODY_XY] = np.clip(self.offset_of(self.target), -self.offset_clip, self.offset_clip)
            pol.command = cmd
            pol.ort_session = self.pick_sess
            self.sim.beak.set_open(1.0 if P.OPEN_START <= ph < P.CLOSE_PHASE else 0.0)
            pol.apply_action(pol.infer())
            self.phase = ph + 0.02 / P.GP_PERIOD
            if self.phase >= 1.0:              # cycle done: hand back to the gait, beak stays shut
                self.phase = None
                pol.vel_cmd = np.zeros(3, dtype=np.float32)
                pol.current_policy = "standing"
                pol.ort_session = pol.standing_session
                pol._update_command()
        else:
            if not np.allclose(pol.vel_cmd, vel_cmd):
                pol.set_vel_cmd(*vel_cmd)         # (upstream prints on every call)
            pol.update_ground_pick_phase(0.02)
            pol.update_behavior(0.02)
            pol.apply_action(pol.infer())
        self.sim.step(4)

    def trunk_z(self):
        return float(self.d.qpos[self.qa + 2])
