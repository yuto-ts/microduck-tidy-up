"""Articulated beak for the Microduck MuJoCo model.

The upstream training model (microduck_rl) welds the lower jaw to the head: the real
robot's 15th servo (`mouth`) is driven outside the RL policy, so the sim omits it.
This module rebuilds the model at load time with the lower jaw split off into its own
body on a hinge, WITHOUT touching the upstream XML files.

Design choices
- The hinge is NOT given a MuJoCo actuator. Upstream inference code sizes the policy
  from `model.nu` (14), so a 15th actuator would break every existing ONNX policy.
  Instead `BeakServo` applies a torque-limited PD law through `qfrc_applied`, the
  same split the real robot uses (policy drives 14 joints, the daemon drives the mouth).
- The joint is named `passive_mouth`: every mjlab selector excludes `passive_*`, so the
  beak stays out of the 61-D observation / 14-D action contract if this model is ever
  used for training.
- In qpos the joint lands right after `head_roll`, matching the real robot's wire order.
"""
from __future__ import annotations

import numpy as np
import mujoco

JAW_PARENT = "jaw_soft"            # head body (child of head_roll)
JAW_BODY = "beak_lower"
JAW_JOINT = "passive_mouth"
# Hinge, in the jaw_soft body frame: the mouth servo's output shaft and the bearing on
# the opposite side line up on this lateral axis (measured from the CAD meshes).
HINGE_POS = np.array([0.0031, 0.0, -0.0180])
HINGE_AXIS = np.array([0.0, 1.0, 0.0])   # +angle opens (lower beak tip goes down)
MOUTH_RANGE = (0.0, 0.70)                # rad; 0 = closed as in the CAD
JAW_MESHES = ("jaw", "jaw_soft")         # lower beak shell + its soft lining
JAW_MASS = 0.008                         # kg, moved out of the head so total mass is unchanged
# Mouth servo law, shared by the training actuator and the CPU-side BeakServo. Torque is
# capped well below an XL330 stall torque so a grasp squeezes, not crushes.
FRONT_CAMERA = "front_cam"
FRONT_CAMERA_FOVY = 60.0  # deg, vertical (the real head camera's lens is wider than 45)
SERVO_KP = 0.6          # N*m/rad
SERVO_KD = 0.01         # N*m*s/rad
SERVO_MAX_TORQUE = 0.25  # N*m

# Collision bits. Robot and floor geoms use contype=conaffinity=1. Props get their own
# bit (4) so that a few robot geoms can be made to ignore them:
#   prop  contype=1|4, conaffinity=4 -> hits floor, walls, robot, beak pads, other props
#   bottom_head_shell conaffinity=0  -> still hits floor/walls, but NOT props
PROP_CONTYPE = 5
PROP_CONAFFINITY = 4

_GEOM_ATTRS = ("type", "meshname", "material", "rgba", "group", "contype", "conaffinity",
               "condim", "friction", "solref", "solimp", "margin", "gap", "priority",
               "size", "quat")


def _copy_geom(src, dst_body, offset):
    g = dst_body.add_geom()
    for a in _GEOM_ATTRS:
        setattr(g, a, getattr(src, a))
    g.pos = np.asarray(src.pos) - offset
    g.name = src.name
    return g


def add_beak(spec: mujoco.MjSpec, pad_friction=(1.5, 0.02, 0.002), actuator: bool = False,
             ) -> mujoco.MjSpec:
    """Split the lower jaw into a hinged body. Mutates and returns `spec`.

    actuator=True also adds a MuJoCo position actuator on `passive_mouth` (named the same).
    Use it for GPU training (mjlab wraps it with XmlActuatorCfg and the task drives its
    target by phase); leave it False for the CPU inference path, where upstream code
    sizes the policy from `model.nu` and `BeakServo` drives the jaw instead.
    """
    head = spec.body(JAW_PARENT)
    if head is None:
        raise ValueError(f"body {JAW_PARENT!r} not found — not a Microduck spec?")
    nq_before = len(spec.keys[0].qpos) if spec.keys else None
    # index of the new dof in qpos = right after head_roll
    head_roll_adr = _qpos_index_after(spec, "head_roll")

    jaw = head.add_body(name=JAW_BODY, pos=HINGE_POS)
    jaw.add_joint(name=JAW_JOINT, type=mujoco.mjtJoint.mjJNT_HINGE, axis=HINGE_AXIS,
                  range=MOUTH_RANGE, limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                  armature=0.002, damping=0.01, frictionloss=0.002)
    # inertia: thin plate ~60 x 90 x 10 mm, CoM about mid-beak, below/behind the hinge
    jaw.mass = JAW_MASS
    jaw.ipos = np.array([-0.0157, 0.0, -0.031])
    jaw.inertia = np.array([6e-6, 3e-6, 6e-6])
    jaw.explicitinertial = True
    head.mass = head.mass - JAW_MASS

    moved = []
    for g in list(head.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname in JAW_MESHES:
            if g.contype == 0 and g.conaffinity == 0:
                _copy_geom(g, jaw, HINGE_POS)        # visual mesh moves with the jaw
            moved.append(g)
    for g in moved:
        spec.delete(g)

    # The bottom head shell's convex hull bulges into the mouth cavity; squeezing a prop
    # against that slope wedges it out of the beak. Keep the shell for floor/wall contact
    # (its contype still matches their conaffinity) but let props pass through it.
    for g in head.geoms:
        if g.meshname == "bottom_head_shell" and (g.contype or g.conaffinity):
            g.conaffinity = 0

    # Collision: the jaw mesh's convex hull would fill the whole mouth cavity, so the
    # beak touches objects through thin box pads on the inner faces instead.
    # Frames: body x = world up, body z = world back (at the STAND pose).
    # Lower pad: inner face of the lower beak (world z ~217-221 mm, x 40-68 mm).
    _pad(jaw, "beak_lower_pad_collision", center_world_rel=(54.0, 0.0, 219.0), half=(14.0, 22.0, 1.5),
         friction=pad_friction)
    _pad(jaw, "beak_lower_tip_collision", center_world_rel=(67.0, 0.0, 221.0), half=(1.5, 22.0, 4.0),
         friction=pad_friction)
    # Upper pad: underside of the upper beak, fixed to the head.
    _pad(head, "beak_upper_pad_collision", center_world_rel=(54.0, 0.0, 225.5), half=(14.0, 22.0, 1.5),
         friction=pad_friction, on_head=True)

    # Head front proxies. When the beak points straight down, the top-shell lip and the
    # camera lens stick out ~4-6 mm past the beak tip and touch the floor first. mjlab's
    # CollisionCfg drops every unnamed mesh geom, so training needs named stand-ins
    # (in CPU MuJoCo they simply overlap the real meshes).
    _pad(head, "head_lip_collision", center_world_rel=(73.0, 0.0, 270.0), half=(2.2, 40.0, 4.0),
         friction=(1.0, 0.005, 0.0001), on_head=True)
    _pad(head, "head_lens_collision", center_world_rel=(70.5, 0.0, 248.0), half=(2.3, 9.0, 8.5),
         friction=(1.0, 0.005, 0.0001), on_head=True)

    # Where a grasped prop's centre sits: between the pads, ~10 mm off the upper pad toward
    # the jaw, ~12 mm back from the tip (head frame, fixed to the head).
    head.add_site(name="beak_grasp", pos=_world_to_head((56.0, 0.0, 214.0)), group=3,
                  size=[0.003, 0, 0], rgba=[1, 1, 0, 1])

    # Forward-looking camera at the lens. The upstream `head_camera` sits inside the head
    # shell and looks backward (export orientation bug), so it renders the shell interior.
    # Camera looks along its -z: here body +z = world -x at STAND, i.e. forward; up = body +x.
    head.add_camera(name=FRONT_CAMERA, pos=_world_to_head((76.0, 0.0, 248.0)),
                    quat=[np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)], fovy=FRONT_CAMERA_FOVY)

    if actuator:
        act = spec.add_actuator(name=JAW_JOINT, target=JAW_JOINT,
                                trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=SERVO_KP, kv=SERVO_KD)
        act.forcelimited = True
        act.forcerange = np.array([-SERVO_MAX_TORQUE, SERVO_MAX_TORQUE])
        act.ctrllimited = True
        act.ctrlrange = np.array(MOUTH_RANGE)

    for key in spec.keys:
        q = list(key.qpos)
        if nq_before is not None and len(q) == nq_before:
            q.insert(head_roll_adr + 1, 0.0)
            key.qpos = q
        if actuator and len(key.ctrl):
            key.ctrl = list(key.ctrl) + [0.0]   # the mouth actuator is appended last
    return spec


# STAND-pose frame of the jaw_soft body (measured): origin in world mm and the map
# world -> body: body_x = world_z, body_y = world_y, body_z = -world_x.
_HEAD_ORIGIN_MM = np.array([-9.0, 0.0, 232.6])


def _world_to_head(p_mm):
    d = (np.asarray(p_mm, float) - _HEAD_ORIGIN_MM) / 1000.0
    return np.array([d[2], d[1], -d[0]])


def _pad(body, name, center_world_rel, half, friction, on_head=False):
    c = _world_to_head(center_world_rel)
    if not on_head:
        c = c - HINGE_POS
    hx, hy, hz = half                       # given in world axes (x fwd, y left, z up), mm
    g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=c,
                      size=np.array([hz, hy, hx]) / 1000.0)   # body x=world z, body z=world x
    g.contype = 1
    g.conaffinity = 1
    g.condim = 4
    g.friction = np.array(friction)
    g.group = 3
    g.priority = 1
    g.rgba = np.array([0.2, 0.8, 0.2, 0.6])
    return g


def _qpos_index_after(spec, joint_name):
    """qpos address of `joint_name` in the compiled model (before adding the beak)."""
    m = spec.copy().compile() if hasattr(spec, "copy") else spec.compile()
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(m.jnt_qposadr[j])


def load_spec_with_beak(xml_path: str) -> mujoco.MjSpec:
    return add_beak(mujoco.MjSpec.from_file(xml_path))


class BeakServo:
    """Scripted mouth servo: torque-limited PD on `passive_mouth` via qfrc_applied.

    `open` is 0..1 like the real robot's mouth command (robotctl / JSON-RPC `RobotMouth`).
    Torque is capped well below an XL330 stall torque so a grasp squeezes, not crushes.
    """

    def __init__(self, model, data, kp=SERVO_KP, kd=SERVO_KD, max_torque=SERVO_MAX_TORQUE):
        self.m, self.d = model, data
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JAW_JOINT)
        if j < 0:
            raise ValueError("model has no beak joint; build it with add_beak()")
        self.qadr = int(model.jnt_qposadr[j])
        self.vadr = int(model.jnt_dofadr[j])
        self.lo, self.hi = model.jnt_range[j]
        self.kp, self.kd, self.max_torque = kp, kd, max_torque
        self.target = 0.0

    def set_open(self, open_frac: float):
        self.target = self.lo + float(np.clip(open_frac, 0.0, 1.0)) * (self.hi - self.lo)

    @property
    def angle(self):
        return float(self.d.qpos[self.qadr])

    def update(self):
        """Call once per physics substep, before mj_step."""
        q, qd = self.d.qpos[self.qadr], self.d.qvel[self.vadr]
        tau = self.kp * (self.target - q) - self.kd * qd
        self.d.qfrc_applied[self.vadr] = np.clip(tau, -self.max_torque, self.max_torque)
