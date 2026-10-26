"""Load the Microduck scene with the articulated beak, BAM servos, and optional props.

Mirrors `load_mujoco_with_bam` in microduck_rl/scripts/infer_policy.py (the path every
upstream ONNX policy is rehearsed on), with the beak spliced in first. Upstream files
are imported, never edited.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import mujoco

from .beak import add_beak, BeakServo, PROP_CONTYPE, PROP_CONAFFINITY

UPSTREAM = os.environ.get(
    "MICRODUCK_RL", os.path.join(os.path.dirname(__file__), "..", "..", "..", "microduck_rl"))
UPSTREAM = os.path.abspath(UPSTREAM)
ROBOT_DIR = os.path.join(UPSTREAM, "src", "mjlab_microduck", "robot", "microduck")
SCENE_FLAT = os.path.join(ROBOT_DIR, "scene.xml")
SCENE_APARTMENT = os.path.join(ROBOT_DIR, "scene_apartment.xml")


def upstream_infer():
    """Import upstream scripts/infer_policy.py as a module."""
    p = os.path.join(UPSTREAM, "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)
    import infer_policy  # noqa: E402
    return infer_policy


@dataclass
class Prop:
    """A free-floating object to pick up."""
    name: str
    pos: tuple
    size: object = 0.020         # cube edge (float) or full box extents (x, y, z), m
    mass: float = 0.010          # kg
    rgba: tuple = (0.9, 0.2, 0.2, 1.0)
    friction: tuple = (1.0, 0.02, 0.002)


@dataclass
class Sim:
    model: mujoco.MjModel
    data: mujoco.MjData
    bam: object
    beak: BeakServo
    props: dict = field(default_factory=dict)   # name -> qpos address

    def step(self, n_substeps: int):
        for _ in range(n_substeps):
            if self.bam is not None:
                self.bam.update()
            self.beak.update()
            mujoco.mj_step(self.model, self.data)


def build(scene=SCENE_FLAT, props=(), dt=0.005, vin=7.4, vin_drop_gain=0.1, bam=True,
          catalog=(), extra=None) -> Sim:
    """props: simple `Prop` boxes. catalog: (name, PropSpec, (x, y), yaw) tuples from
    `microduck_tidy.props` (cubes, cylinders, spheres with real sizes/masses)."""
    ip = upstream_infer()
    spec = add_beak(mujoco.MjSpec.from_file(scene))
    for p in props:
        b = spec.worldbody.add_body(name=p.name, pos=list(p.pos))
        b.add_freejoint(name=f"{p.name}_free")
        half = [p.size / 2] * 3 if np.isscalar(p.size) else [e / 2 for e in p.size]
        b.add_geom(name=f"{p.name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=half,
                   mass=p.mass, rgba=list(p.rgba), condim=4, friction=list(p.friction),
                   contype=PROP_CONTYPE, conaffinity=PROP_CONAFFINITY)
    if extra is not None:          # e.g. Tray.add_to: static scenery, no qpos
        extra(spec)
    cat_pose = []
    for name, ps, xy, yaw in catalog:
        b = ps.add_to(spec.worldbody, name, xy, contype=PROP_CONTYPE,
                      conaffinity=PROP_CONAFFINITY, yaw=yaw)
        cat_pose.append(list(b.pos) + list(b.quat))
    # Keyframes hold the robot only; props were appended after, so pad them out.
    for key in spec.keys:
        if props or catalog:
            q = list(key.qpos)
            for p in props:
                q += list(p.pos) + [1, 0, 0, 0]
            for pose in cat_pose:
                q += pose
            key.qpos = q

    bam_ctrl = None
    if bam:
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, vin, None)
        kt, R = bam_model.kt.value, bam_model.R.value
        force_limit = bam_model.actuator.vin * kt / R
        names = []
        for act in spec.actuators:
            tgt = act.target if isinstance(act.target, str) else act.target.name
            if tgt.startswith("passive_"):
                continue
            act.set_to_motor()
            act.forcelimited = True
            act.forcerange = (-force_limit, force_limit)
            act.ctrllimited = False
            act.gear = [1.0, 0, 0, 0, 0, 0]
            names.append(act.name)
            j = spec.joint(tgt)
            j.damping = np.zeros((3, 1))
            j.frictionloss = 0.0
            j.solref_friction = ip.BAM_STIFF_SOLREF_FRICTION
            j.solimp_friction = ip.BAM_STIFF_SOLIMP_FRICTION
        model = spec.compile()
        model.opt.timestep = dt
        data = mujoco.MjData(model)
        from bam.mujoco import MujocoController
        bam_ctrl = MujocoController(bam_model, names, model, data,
                                    vin_drop_gain=vin_drop_gain, vin_min=ip.BAM_VIN_MIN)
    else:
        model = spec.compile()
        model.opt.timestep = dt
        data = mujoco.MjData(model)
    prop_adr = {n: int(model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_free")]) for n in
        [p.name for p in props] + [c[0] for c in catalog]}
    return Sim(model, data, bam_ctrl, BeakServo(model, data), prop_adr)


def _pad_midpoint(sim: Sim):
    m, d = sim.model, sim.data
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    return 0.5 * (d.geom_xpos[gid("beak_lower_pad_collision")] + d.geom_xpos[gid("beak_upper_pad_collision")])


def place_in_beak(sim: Sim, prop: str, opening: float = 0.45):
    """Hand `prop` to the duck: open the beak to `opening` rad, park the prop between the
    pads, and command the beak shut. Call `hold_in_gap` for ~0.2 s afterwards so the
    prop stays put while the jaw travels onto it (a hand-over, not a teleport into
    penetration: a 20 mm cube does not fit the gap below ~0.4 rad)."""
    m, d = sim.model, sim.data
    d.qpos[sim.beak.qadr] = opening
    mujoco.mj_forward(m, d)
    sim.beak.set_open(0.0)
    hold_in_gap(sim, prop)


def hold_in_gap(sim: Sim, prop: str):
    m, d = sim.model, sim.data
    jb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "beak_lower")
    a = sim.props[prop]
    v = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prop}_free")])
    d.qpos[a:a + 3] = _pad_midpoint(sim)
    d.qpos[a + 3:a + 7] = d.xquat[jb]
    d.qvel[v:v + 6] = 0.0
    mujoco.mj_forward(m, d)
