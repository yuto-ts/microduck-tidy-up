"""Search -> approach -> pick -> carry -> drop, on top of the learned policies.

Layers (what runs where):
- `FrontCamPerception`: renders the head's forward camera (segmentation + depth) and
  turns the pixels of each object into a world position. This stands in for an object
  detector on the real camera image; on the robot the same interface would be filled by
  a detector + the depth sensor.
- `FetchController`: a state machine that sends velocity / head-pitch commands to the
  stock walking policy, triggers the BeakPick policy when the target sits inside the
  policy's trained placement window, carries the object to a tray and opens the beak.
  The robot's world pose stands in for the robot's odometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import pick_profile as P
from .beak import FRONT_CAMERA


# -- perception ---------------------------------------------------------------------------
@dataclass
class Detection:
    name: str
    world: np.ndarray        # estimated object centre (x, y, z)
    pixels: int
    bbox: tuple              # (u0, v0, u1, v1) in the camera image


class FrontCamPerception:
    def __init__(self, model, data, targets: dict, width=160, height=120, min_pixels=4):
        """targets: {object name: geom id}."""
        self.m, self.d = model, data
        self.W, self.H = width, height
        self.min_pixels = min_pixels
        self.targets = targets
        self.cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, FRONT_CAMERA)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.r = mujoco.Renderer(model, height, width)
        fovy = math.radians(model.cam_fovy[self.cam])
        self.f = (height / 2) / math.tan(fovy / 2)
        self.last_rgb = None
        self.want_rgb = False             # also keep the RGB frame (for videos)

    def observe(self, want_rgb=None):
        want_rgb = self.want_rgb if want_rgb is None else want_rgb
        r, d = self.r, self.d
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=self.cam)
        seg = r.render().copy()
        r.disable_segmentation_rendering()
        r.enable_depth_rendering()
        r.update_scene(d, camera=self.cam)
        depth = r.render().copy()
        r.disable_depth_rendering()
        if want_rgb:
            r.update_scene(d, camera=self.cam)
            self.last_rgb = r.render().copy()
        geom_ids = np.where(seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[:, :, 0], -1)
        cam_pos = d.cam_xpos[self.cam].copy()
        cam_R = d.cam_xmat[self.cam].reshape(3, 3).copy()
        out = []
        for name, gid in self.targets.items():
            vs, us = np.nonzero(geom_ids == gid)
            if len(vs) < self.min_pixels:
                continue
            u, v = us.mean(), vs.mean()
            z = float(np.median(depth[vs, us]))
            pc = np.array([(u + 0.5 - self.W / 2) / self.f * z,
                           -(v + 0.5 - self.H / 2) / self.f * z, -z])
            p = cam_pos + cam_R @ pc                          # visible front surface
            ray = p - cam_pos; ray[2] = 0.0
            n = np.linalg.norm(ray)
            if n > 1e-6:
                p = p + ray / n * 0.008                       # ~half an object deeper
            out.append(Detection(name, p, len(vs), (us.min(), vs.min(), us.max(), vs.max())))
        return out


# -- controller ---------------------------------------------------------------------------
@dataclass
class Tray:
    """Collection area: a flat mat on the floor (no walls: the feet trip on anything
    taller than ~1 cm, and the robot has to stand right at its edge to drop into it)."""
    center: tuple = (-0.40, -0.45)
    size: float = 0.22

    def add_to(self, spec: mujoco.MjSpec):
        b = spec.worldbody.add_body(name="tray", pos=[self.center[0], self.center[1], 0.0])
        h = self.size / 2
        b.add_geom(name="tray_mat", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0.0005],
                   size=[h, h, 0.0005], rgba=[0.95, 0.75, 0.2, 1], contype=0, conaffinity=0)
        b.add_geom(name="tray_inner", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0.0008],
                   size=[h - 0.012, h - 0.012, 0.0005], rgba=[0.98, 0.92, 0.7, 1],
                   contype=0, conaffinity=0)

    def contains(self, xy) -> bool:
        return (abs(xy[0] - self.center[0]) < self.size / 2
                and abs(xy[1] - self.center[1]) < self.size / 2)


@dataclass
class FetchParams:
    # Where to put the object relative to the nominal pick spot, and the offsets the pick
    # policy actually succeeds at (measured with scripts/pick_window_map.py; v2_1000:
    # dx -30..+15 mm, dy -40..0 mm reliable), robot yaw frame.
    aim_dx: float = -0.008
    aim_dy: float = -0.015
    win_x: tuple = (-0.028, 0.014)
    win_y: tuple = (-0.035, 0.004)
    head_far: float = 0.45               # head_pitch command while searching / far (rad)
    head_near: float = 0.85              # ... when the target is close
    near_dist: float = 0.35
    turn_rate: float = 0.6
    turn_in_place: bool = False          # the walking policies cannot spin on the spot:
                                         # turn while walking slowly instead
    arc_speed: float = 0.25
    walk_fast: float = 0.30
    walk_slow: float = 0.30              # 0.25 is at the walk/stand boundary (often no step)
    back_speed: float = 0.35
    final_dist: float = 0.30             # switch to the slow, steered final approach
    step_s: float = 0.35                 # forward nudge: ~2-4 cm
    back_s: float = 1.0                  # back-off before re-approaching: ~13 cm
    stop_ahead: float = 0.035            # stop when the goal is this close (m)
    arena_radius: float = 1.3
    perceive_every: int = 5              # control steps (100 ms)


@dataclass
class FetchController:
    runner: object                       # PickRunner
    perception: FrontCamPerception
    tray: Tray
    params: FetchParams = field(default_factory=FetchParams)

    def __post_init__(self):
        self.state = "SEARCH"
        self.t_state = 0.0
        self.memory = {}                 # name -> estimated world xy (last seen)
        self.target = None
        self.delivered = set()
        self.failed = {}                 # name -> pick attempts that did not lift it
        self.turned = 0.0
        self.last_yaw = None
        self.k = 0
        self.log = []
        self.detections = []

    # -- helpers ---------------------------------------------------------------------------
    def _rel(self, xy):
        x0, y0, yaw = self.runner.root_xy_yaw()
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = xy[0] - x0, xy[1] - y0
        return c * dx + s * dy, -s * dx + c * dy

    def _set(self, state):
        if state != self.state:
            self.log.append((round(self.k * 0.02, 2), state, self.target))
        self.state, self.t_state = state, 0.0

    def _grasp_rel(self):
        """Beak grasp point in the robot yaw frame (forward, left), from kinematics."""
        sid = self.runner.grasp_site
        p = self.runner.d.site_xpos[sid]
        return self._rel((p[0], p[1]))

    def _head(self, pitch):
        self.runner.pol.head_offset = np.array([0.0, pitch, 0.0, 0.0], dtype=np.float32)

    def _candidates(self):
        out = {}
        for n, xy in self.memory.items():
            if n in self.delivered or self.failed.get(n, 0) >= 3:
                continue
            if self.tray.contains(xy):
                continue
            out[n] = xy
        return out

    def _nav(self, goal_xy, ahead):
        """Velocity command that brings `goal_xy` to (ahead, 0) in the robot frame.
        Returns (cmd, arrived)."""
        rx, ry = self._rel(goal_xy)
        bearing = math.atan2(ry, rx)
        ex = rx - ahead
        dist = math.hypot(rx, ry)
        p = self.params
        if ex <= p.stop_ahead and abs(ry) < 0.05:
            return (0.0, 0.0, 0.0), True
        if dist > 0.15 and abs(bearing) > 0.35:
            w = float(np.clip(1.5 * bearing, -0.8, 0.8))
            if p.turn_in_place:
                return (0.0, 0.0, w), False
            if dist < 0.30:                  # too close to arc round it: back off first
                return (-p.back_speed, 0.0, 0.0), False
            return (p.arc_speed, 0.0, float(np.sign(w)) * max(abs(w), 0.6)), False
        if ex <= p.stop_ahead:              # beside it: back off and come in again
            return (-p.back_speed, 0.0, 0.0), False
        v = p.walk_fast if ex > 0.25 else p.walk_slow
        return (v, 0.0, float(np.clip(2.0 * bearing, -0.6, 0.6))), False

    # -- main step ---------------------------------------------------------------------------
    def step(self):
        p, run = self.params, self.runner
        dt = 0.02
        self.k += 1
        self.t_state += dt

        if run.picking:                   # BeakPick owns the robot until the cycle ends
            run.step()
            return

        if self.k % p.perceive_every == 0 and self.state in ("SEARCH", "WANDER", "APPROACH", "FINAL", "SETTLE", "BACKUP"):
            self.detections = [det for det in self.perception.observe()
                               if det.name not in self.delivered]
            for det in self.detections:
                old = self.memory.get(det.name)
                new = det.world[:2]
                self.memory[det.name] = new if old is None else 0.5 * old + 0.5 * new

        cmd = (0.0, 0.0, 0.0)
        st = self.state
        if st == "SEARCH":
            self._head(p.head_far)
            cands = self._candidates()
            if cands:
                x0, y0, _ = run.root_xy_yaw()
                self.target = min(cands, key=lambda n: math.hypot(cands[n][0] - x0, cands[n][1] - y0))
                self.turned = 0.0
                self._set("APPROACH")
            else:
                _, _, yaw = run.root_xy_yaw()
                if self.last_yaw is not None:
                    self.turned += abs((yaw - self.last_yaw + math.pi) % (2 * math.pi) - math.pi)
                self.last_yaw = yaw
                cmd = ((0.0, 0.0, p.turn_rate) if p.turn_in_place
                       else (p.arc_speed, 0.0, p.turn_rate))   # sweep the floor in a circle
                if self.turned > 2 * math.pi:
                    self.turned = 0.0
                    self._set("WANDER")
        elif st == "WANDER":
            x0, y0, yaw = run.root_xy_yaw()
            if math.hypot(x0, y0) > p.arena_radius * 0.7:   # head back toward the middle
                cmd, _ = self._nav((0.0, 0.0), 0.0)
            else:
                cmd = (p.walk_fast, 0.0, 0.0)
            if self._candidates() or self.t_state > 2.0:
                self.last_yaw = None
                self._set("SEARCH")
        elif st == "APPROACH":
            goal = self.memory[self.target]
            rx, ry = self._rel(goal)
            if math.hypot(rx, ry) < p.final_dist and abs(math.atan2(ry, rx)) < 0.5:
                self._set("FINAL")
            else:
                self._head(p.head_far)
                cmd, _ = self._nav(goal, P.NOMINAL[0])
        elif st == "FINAL":                  # slow, steered so the object ends at the aim point
            self._head(p.head_near)
            rx, ry = self._rel(self.memory[self.target])
            ex = rx - (P.NOMINAL[0] + p.aim_dx)
            ey = ry - (P.NOMINAL[1] + p.aim_dy)
            if ex <= 0.02:
                self._set("SETTLE")
            elif ex < -0.05 or abs(math.atan2(ry, rx)) > 0.8:
                self._set("BACKUP")
            else:
                cmd = (p.walk_slow, 0.0, float(np.clip(2.5 * math.atan2(ey, max(rx, 0.05)), -0.5, 0.5)))
        elif st == "SETTLE":
            self._head(p.head_near)
            if self.t_state > 0.8:
                dx, dy = run.offset_of_xy(self.memory[self.target])
                in_x = p.win_x[0] <= dx <= p.win_x[1]
                in_y = p.win_y[0] <= dy <= p.win_y[1]
                if in_x and in_y:
                    self._head(0.0)
                    run.start_pick(self.target, self.memory[self.target])
                    self._set("PICKING")
                elif in_y and dx > p.win_x[1]:
                    self._set("STEP")
                else:
                    self.retries = getattr(self, "retries", 0) + 1
                    self._set("BACKUP")
        elif st == "STEP":
            cmd = (p.walk_slow, 0.0, 0.0)
            if self.t_state > p.step_s:
                self._set("SETTLE")
        elif st == "BACKUP":
            self._head(p.head_near)
            cmd = (-p.back_speed, 0.0, 0.0)
            if self.t_state > p.back_s:
                self._set("APPROACH" if self.target in self._candidates() else "SEARCH")
        elif st == "PICKING":            # pick cycle just ended (run.picking is False)
            held = run.grasp_dist(self.target) < 0.03 and run.prop_pos(self.target)[2] > 0.04
            if held:
                self._set("CARRY")
            else:
                self.failed[self.target] = self.failed.get(self.target, 0) + 1
                self.memory.pop(self.target, None)       # it has moved: look again
                self._set("BACKUP")
        elif st == "CARRY":
            self._head(0.0)
            # bring the beak (not the robot) over the mat centre: aim the robot so the
            # grasp point's offset (from joint kinematics) lands on the centre
            gx, gy = self._grasp_rel()
            rx, ry = self._rel(self.tray.center)
            cmd, arrived = self._nav(self.tray.center, gx)
            if arrived and abs(ry - gy) < 0.05:
                self._set("DROP")
        elif st == "DROP":
            run.sim.beak.set_open(1.0)
            if self.t_state > 1.0:
                run.sim.beak.set_open(0.0)
                if self.tray.contains(run.prop_pos(self.target)[:2]):
                    self.delivered.add(self.target)
                self.memory.pop(self.target, None)
                self.target = None
                self._set("LEAVE")
        elif st == "LEAVE":
            cmd = (-p.walk_slow, 0.0, 0.0)
            if self.t_state > 1.0:
                self.last_yaw = None
                self._set("SEARCH")
        if self.state not in ("FINAL", "SETTLE", "STEP"):
            cmd = self._avoid_tray(cmd)
        run.step(cmd)

    def _avoid_tray(self, cmd):
        """Steer away from the tray walls except when delivering (feet trip on them)."""
        if self.state in ("CARRY", "DROP") or cmd[0] <= 0.0:
            return cmd
        rx, ry = self._rel(self.tray.center)
        d = math.hypot(rx, ry)
        if d < 0.33 and rx > -0.05:
            if d < 0.22:
                return (-self.params.walk_slow, 0.0, 0.0)
            return (self.params.arc_speed, 0.0, -0.8 if ry > 0 else 0.8)
        return cmd