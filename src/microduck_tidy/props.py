"""Everyday objects a Microduck could plausibly pick up with its beak.

Geometry the beak imposes (measured on the model):
- pads are ~44 mm wide and ~28 mm long; the opening at the pads is ~29 mm at full open
  (0.70 rad), and a 15-20 mm block is held reliably at a partial close
- with the beak pointing down, the tip stops a few mm above the floor (the head's top lip
  and lens stick out past it), so the part to be gripped must stand at least ~12 mm tall
- the mouth servo is torque-capped at 0.25 N*m and objects of 10-30 g were carried while
  walking

Each entry is a MuJoCo primitive stand-in with a real-world size and mass. `GRASPABLE` is
the candidate list; `BORDERLINE` are objects worth showing in the scene as limits (too low,
too wide, or too heavy) — useful for the report's discussion.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class PropSpec:
    key: str
    name_ja: str
    shape: str                 # box | cylinder | sphere | capsule
    size: tuple                # box: full (x, y, z); cylinder: (diameter, height); sphere: (diameter,)
    mass: float                # kg
    rgba: tuple
    friction: float = 1.0
    lying: bool = False        # cylinder/capsule lying on its side, axis along robot y
    note: str = ""

    @property
    def height(self) -> float:
        if self.shape == "box":
            return self.size[2]
        if self.shape == "sphere":
            return self.size[0]
        return self.size[0] if self.lying else self.size[1]

    def add_to(self, parent: mujoco.MjsBody, name: str, pos, contype=5, conaffinity=4,
               freejoint=True, yaw=0.0) -> mujoco.MjsBody:
        """Add as a free body. `pos` is (x, y) on the floor; z is set from the height."""
        q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        b = parent.add_body(name=name, pos=[pos[0], pos[1], self.height / 2 + 1e-4], quat=q)
        if freejoint:
            b.add_freejoint(name=f"{name}_free")
        g = b.add_geom(name=f"{name}_geom", mass=self.mass, rgba=list(self.rgba), condim=4,
                       friction=[self.friction, 0.01, 0.0005],
                       contype=contype, conaffinity=conaffinity)
        if self.shape == "box":
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [s / 2 for s in self.size]
        elif self.shape == "sphere":
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size = [self.size[0] / 2, 0, 0]
        else:
            g.type = (mujoco.mjtGeom.mjGEOM_CYLINDER if self.shape == "cylinder"
                      else mujoco.mjtGeom.mjGEOM_CAPSULE)
            r, h = self.size[0] / 2, self.size[1] / 2
            if self.shape == "capsule":
                h = max(h - r, 1e-4)
            g.size = [r, h, 0]
            if self.lying:   # axis along local y
                g.quat = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
        return b


GRASPABLE = [
    PropSpec("wood_cube_20", "木製キューブ 20mm", "box", (0.020, 0.020, 0.020), 0.006,
             (0.85, 0.65, 0.40, 1), 0.8, note="学習対象。積み木の基本サイズ"),
    PropSpec("dice_16", "サイコロ 16mm", "box", (0.016, 0.016, 0.016), 0.004,
             (0.95, 0.95, 0.95, 1), 0.6, note="小さく軽い。高さぎりぎり"),
    PropSpec("cork", "ワインコルク(立て)", "cylinder", (0.024, 0.044), 0.004,
             (0.75, 0.58, 0.38, 1), 1.0, note="背が高く挟みやすい"),
    PropSpec("battery_aa", "単3電池(立て)", "cylinder", (0.0145, 0.0505), 0.023,
             (0.15, 0.15, 0.15, 1), 0.5, note="細く重い。滑りやすい"),
    PropSpec("sponge", "キッチンスポンジ片", "box", (0.022, 0.030, 0.025), 0.002,
             (0.95, 0.85, 0.20, 1), 1.2, note="軽く摩擦が大きい(剛体で近似)"),
    PropSpec("paper_ball", "丸めた紙", "sphere", (0.024,), 0.002,
             (0.92, 0.92, 0.88, 1), 0.9, note="球なので転がりやすい"),
    PropSpec("eraser_side", "消しゴム(横置き)", "box", (0.011, 0.043, 0.017), 0.010,
             (0.30, 0.45, 0.85, 1), 1.1, note="薄い面を挟む向き"),
    PropSpec("lego_2x2_tall", "LEGO 2x2 ブロック2段", "box", (0.016, 0.016, 0.0192), 0.005,
             (0.85, 0.10, 0.10, 1), 0.7, note="2 段重ねで高さを稼ぐ"),
    PropSpec("mini_car", "ミニカー", "box", (0.018, 0.040, 0.015), 0.020,
             (0.10, 0.55, 0.30, 1), 0.7, note="横向きに置いた状態"),
    PropSpec("marker_cap", "ペンのキャップ(立て)", "cylinder", (0.015, 0.035), 0.002,
             (0.10, 0.25, 0.80, 1), 0.6, note="軽く細い"),
]

BORDERLINE = [
    PropSpec("bottle_cap", "ペットボトルキャップ", "cylinder", (0.030, 0.012), 0.002,
             (0.95, 0.95, 0.95, 1), 0.7, note="低すぎて先端が届かない見込み"),
    PropSpec("lego_2x4", "LEGO 2x4 ブロック", "box", (0.016, 0.032, 0.0115), 0.0023,
             (0.95, 0.80, 0.10, 1), 0.7, note="1 段だと低すぎる"),
    PropSpec("pingpong", "ピンポン球 40mm", "sphere", (0.040,), 0.0027,
             (1.0, 0.55, 0.10, 1), 0.5, note="開口より大きい"),
    PropSpec("big_cube_30", "積み木 30mm", "box", (0.030, 0.030, 0.030), 0.018,
             (0.60, 0.40, 0.25, 1), 0.8, note="開口ぎりぎり、固定把持試験では落下"),
]

CATALOG = {p.key: p for p in GRASPABLE + BORDERLINE}
