"""BeakPick: crouch, open the beak around a small object on the floor, shut it, stand up
holding it.

Built on upstream's ground-pick task (same phase command, return-pose and stability
terms, same 61-D actor observation / 14-D action contract) with three additions:

1. Robot model with an articulated beak (`microduck_tidy.beak.add_beak`). The mouth is a
   15th MuJoCo actuator that the POLICY DOES NOT DRIVE: the action term is restricted to
   the 14 servo joints and a per-step hook opens/closes the beak by phase, the way the real
   robot's daemon commands the mouth outside the policy.
2. A free 20 mm cube ("prop") in front of the feet, re-placed every episode with
   +-15/20 mm noise and random yaw.
3. The prop's offset from the nominal pick spot is fed to the actor through the
   body_pose command slot (zero-padded upstream), so the policy can aim the beak. The
   critic additionally sees the prop's position/velocity and the beak opening.

Phase profile (period GP_PERIOD):
  [0, DESCENT_END)          crouch, beak opens at OPEN_START
  [DESCENT_END, HOLD_END)   at the bottom; beak shuts at CLOSE_PHASE
  [HOLD_END, RISE_END)      stand back up holding the prop
  [RISE_END, 1)             stand still, prop in the beak
"""
from __future__ import annotations

import math

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab_microduck.robot.microduck_constants import (
    HOME_FRAME,
    MICRODUCK_GROUNDCONTACT_XML,
    actuators as BAM_ACTUATORS,
)
from mjlab_microduck.tasks import mdp as upstream_mdp
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

from microduck_tidy.beak import add_beak, PROP_CONTYPE, PROP_CONAFFINITY
from microduck_tidy.tasks import mdp

# -- phase profile / pick spot (shared with the CPU evaluation runner) --------------------
from microduck_tidy.pick_profile import (  # noqa: E402
    GP_PERIOD, DESCENT_END, HOLD_END, RISE_END, OPEN_START, CLOSE_PHASE, OPEN_ANGLE,
    NOMINAL, NOISE_X, NOISE_Y, OFFSET_CLIP,
)

# -- prop ---------------------------------------------------------------------------------
PROP_SIZE = 0.020                     # cube edge (m)
PROP_MASS = 0.010                     # kg (DR'd at startup)
PROP_REST_Z = PROP_SIZE / 2


def _robot_spec() -> mujoco.MjSpec:
    return add_beak(mujoco.MjSpec.from_file(str(MICRODUCK_GROUNDCONTACT_XML)), actuator=True)


def _prop_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(name="prop", pos=[0.3, 0.0, PROP_REST_Z])
    b.add_freejoint(name="prop_free")
    b.add_geom(name="prop_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[PROP_SIZE / 2] * 3,
               mass=PROP_MASS, rgba=[0.85, 0.65, 0.40, 1], condim=4,
               friction=[0.8, 0.01, 0.0005], contype=PROP_CONTYPE,
               conaffinity=PROP_CONAFFINITY)
    return spec


_FEET = r"^(left|right)_foot_collision$"
_BEAK = r"^beak_.*_collision$"
BEAK_PICK_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_FEET: 3, _BEAK: 4, r"^head_.*_collision$": 3, ".*_collision": 1},
    priority={_FEET: 1, _BEAK: 1},
    friction={_FEET: (1.0,), _BEAK: (1.5, 0.02, 0.002)},
)

BEAK_PICK_ROBOT_CFG = EntityCfg(
    spec_fn=_robot_spec,
    init_state=EntityCfg.InitialStateCfg(
        joint_pos={r"^passive_mouth$": 0.0, **HOME_FRAME.joint_pos},
        joint_vel={".*": 0.0},
    ),
    collisions=(BEAK_PICK_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(BAM_ACTUATORS, XmlActuatorCfg(target_names_expr=(r"^passive_mouth$",))),
        soft_joint_pos_limit_factor=0.9,
    ),
)

PROP_CFG = EntityCfg(
    spec_fn=_prop_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, PROP_REST_Z)),
)


def make_beak_pick_env_cfg(play: bool = False, noise_x: float = NOISE_X,
                           noise_y: float = NOISE_Y, yaw_range: float = 0.6,
                           offset_clip: float = OFFSET_CLIP) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_ground_pick_env_cfg(play=play)

    # -- scene: beak robot (must stay first) + prop
    cfg.scene.entities = {"robot": BEAK_PICK_ROBOT_CFG, "prop": PROP_CFG}
    cfg.sim.nconmax = 60
    cfg.episode_length_s = 2 * GP_PERIOD     # >= one full pick cycle from any start phase

    # -- actions: the 14 servos only; the mouth actuator belongs to the phase hook
    act = cfg.actions["joint_pos"]
    assert isinstance(act, JointPositionActionCfg)
    act.actuator_names = (r"^(?!passive_).*",)

    # -- command: same phase encoding, longer period
    cfg.commands["twist"].period = GP_PERIOD

    # SceneEntityCfg params must be passed explicitly: the managers only resolve names
    # (site/joint ids) for configs that appear in a term's params.
    grasp = SceneEntityCfg("robot", site_names=["beak_grasp"])
    mouth = SceneEntityCfg("robot", joint_names=["passive_mouth"])
    gates = dict(descent_end=DESCENT_END, hold_end=HOLD_END, rise_end=RISE_END,
                 asset_cfg=grasp)
    up_gates = dict(hold_end=HOLD_END, rise_end=RISE_END)

    # -- rewards: drop the "mouth near the floor, never touch" objective and the
    #    simulated payload; the prop is real now
    cfg.rewards.pop("mouth_payload_force", None)
    # Keep upstream's "mouth tip toward the floor" pull as crouch shaping (it is what makes
    # the stock policy discover the crouch), lighter than upstream's 3.0: the grasp terms
    # below take over once the beak is near the prop.
    cfg.rewards["mouth_ground_proximity"].params.update(
        descent_end=DESCENT_END, hold_end=HOLD_END, rise_end=RISE_END)
    cfg.rewards["mouth_ground_proximity"].weight = 1.0
    cfg.events.pop("sample_mouth_payload", None)

    for name in ("mouth_perpendicular_to_ground",):
        cfg.rewards[name].params.update(descent_end=DESCENT_END, hold_end=HOLD_END,
                                        rise_end=RISE_END)
        cfg.rewards[name].weight = 1.0
    for name in ("ground_pick_return_pose_legs", "ground_pick_return_pose_neck",
                 "return_upright"):
        cfg.rewards[name].params.update(up_gates)
    cfg.rewards["neck_vel_descent"].params["hold_end"] = HOLD_END
    # The beak has to touch the floor lightly to reach around the prop: tolerate light
    # head contact, still punish slamming.
    cfg.rewards["head_impact_penalty"].weight = -1.0
    cfg.rewards["head_impact_penalty"].params["threshold"] = 3.0

    # NB: mjlab's RewardManager skips weight == 0.0 terms entirely (never calls them), so
    # hooks need a non-zero weight. The hook returns zeros; the weight is cosmetic.
    cfg.rewards["beak_control"] = RewardTermCfg(
        func=mdp.beak_phase_control, weight=1e-6,
        params={"open_start": OPEN_START, "close_phase": CLOSE_PHASE, "open_angle": OPEN_ANGLE,
                "asset_cfg": mouth},
    )
    # Three widths: standing, the grasp point is ~0.19 m from the prop, so the coarse
    # kernel must still have gradient there (std 0.15 -> ~0.2); mid and fine take over
    # near the target.
    cfg.rewards["grasp_approach_coarse"] = RewardTermCfg(
        func=mdp.grasp_approach, weight=2.0, params={"std": 0.15, **gates})
    cfg.rewards["grasp_approach_mid"] = RewardTermCfg(
        func=mdp.grasp_approach, weight=3.0, params={"std": 0.04, **gates})
    cfg.rewards["grasp_approach_fine"] = RewardTermCfg(
        func=mdp.grasp_approach, weight=4.0, params={"std": 0.012, **gates})
    cfg.rewards["prop_held"] = RewardTermCfg(
        func=mdp.prop_held, weight=4.0, params={"std": 0.02, "asset_cfg": grasp, **up_gates})
    cfg.rewards["prop_lifted"] = RewardTermCfg(
        func=mdp.prop_lifted, weight=8.0,
        params={"lift_height": 0.06, "hold_std": 0.025, "rest_z": PROP_REST_Z,
                "asset_cfg": grasp, **up_gates})
    cfg.rewards["prop_knocked"] = RewardTermCfg(
        func=mdp.prop_knocked, weight=-1.0, params={"close_phase": CLOSE_PHASE})
    # Success metric; tiny weight so it is computed and logged (Episode_Reward/pick_success
    # = fraction-of-time * weight).
    cfg.rewards["pick_success"] = RewardTermCfg(
        func=mdp.pick_success, weight=1e-3,
        params={"rest_z": PROP_REST_Z, "phase_lo": RISE_END, "asset_cfg": grasp})

    # -- observations: prop offset in the body_command slot (actor + critic), privileged
    #    prop state + beak opening for the critic only
    for grp, noise in (("actor", 0.003), ("critic", 0.0)):
        cfg.observations[grp].terms["body_command"] = ObservationTermCfg(
            func=mdp.prop_offset_command, params={"nominal": NOMINAL, "noise": noise, "clip": offset_clip})
    cfg.observations["critic"].terms["prop_state"] = ObservationTermCfg(
        func=mdp.prop_state_privileged)
    cfg.observations["critic"].terms["mouth_angle"] = ObservationTermCfg(
        func=mdp.mouth_angle, params={"asset_cfg": mouth})

    # -- events: place the prop after the robot reset; DR its mass and friction
    cfg.events["reset_prop"] = EventTermCfg(
        func=mdp.reset_prop_in_front, mode="reset",
        params={"nominal": NOMINAL, "noise_x": noise_x, "noise_y": noise_y,
                "rest_z": PROP_REST_Z, "yaw_range": yaw_range},
    )
    cfg.events["prop_mass"] = EventTermCfg(
        func=dr.body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("prop", body_names=("prop",)),
                "operation": "abs", "ranges": (0.004, 0.030)},
    )
    cfg.events["prop_friction"] = EventTermCfg(
        func=dr.geom_friction, mode="startup",
        params={"asset_cfg": SceneEntityCfg("prop", geom_names=("prop_geom",)),
                "operation": "abs", "ranges": (0.5, 1.2)},
    )
    return cfg


BeakPickRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0,
                          "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu",
                         obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=None,
    ),
    wandb_project="microduck_tidy",
    experiment_name="beak_pick",
    run_name="beak_pick",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6000,
)
