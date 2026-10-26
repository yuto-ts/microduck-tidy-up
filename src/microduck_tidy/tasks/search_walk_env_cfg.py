"""SearchWalk: the upstream velocity task, retargeted to walking while looking at the floor.

Why: searching for objects needs (1) walking with the head pitched down so the front
camera sees the floor, and (2) turning on the spot. Measured on the stock walking
policies: turn-in-place at 0.7 rad/s barely moves (7-10 deg in 5.5 s), and walking with
the head pitched down either stalls or drifts 160-200 deg off heading.

Changes vs `make_microduck_velocity_env_cfg` (same 61-D obs / 14-D action contract):
- head_pose command: fixed ranges biased to looking down (head_pitch -0.1..0.95 rad,
  neck_pitch -0.1..0.4), no range curriculum (the policy is fine-tuned from a trained
  walker that already went through it)
- twist: 30% of envs practise turning in place (upstream 15%)
- yaw-rate tracking tightened (std sqrt(0.5) -> sqrt(0.15), weight 2.0 -> 3.5). With the
  upstream std, standing still under a 0.7 rad/s turn command still earns ~40% of the
  tracking reward, and the sw1 run (upstream std) indeed never learned to turn on the spot.
"""
from __future__ import annotations

import math
from copy import deepcopy

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

HEAD_RANGES = (
    (-0.10, 0.40),    # neck_pitch
    (-0.10, 0.95),    # head_pitch (+ = look down)
    (-0.35, 0.35),    # head_yaw
    (-0.05, 0.05),    # head_roll
)
TURN_IN_PLACE = 0.30
ANG_TRACK_STD = math.sqrt(0.15)
ANG_TRACK_WEIGHT = 3.5


def make_search_walk_env_cfg(play: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play)
    cfg.commands["twist"].rel_turn_in_place_envs = TURN_IN_PLACE
    cfg.commands["head_pose"].ranges = HEAD_RANGES
    cfg.curriculum.pop("head_pose_range", None)
    cfg.rewards["track_angular_velocity"].params["std"] = ANG_TRACK_STD
    cfg.rewards["track_angular_velocity"].weight = ANG_TRACK_WEIGHT
    return cfg


SearchWalkRlCfg = deepcopy(MicroduckRlCfg)
SearchWalkRlCfg.experiment_name = "search_walk"
SearchWalkRlCfg.run_name = "search_walk"
