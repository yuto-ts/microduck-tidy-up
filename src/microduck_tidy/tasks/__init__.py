"""Register microduck_tidy tasks with mjlab's task registry.

Importing this module also imports upstream `mjlab_microduck.tasks` (its registrations
and the shared runner class). The launchers in scripts/ import it before handing over to
mjlab's CLI, so nothing has to be installed into the upstream environment.
"""
from mjlab.tasks.registry import register_mjlab_task

from mjlab_microduck.tasks import MicroduckOnPolicyRunner

from .beak_pick_env_cfg import BeakPickRlCfg, make_beak_pick_env_cfg

register_mjlab_task(
    task_id="Mjlab-BeakPick-Flat-MicroDuck",
    env_cfg=make_beak_pick_env_cfg(),
    play_env_cfg=make_beak_pick_env_cfg(play=True),
    rl_cfg=BeakPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Wide placement window: the pick after walking up to an object. Same obs/action layout
# and experiment folder as BeakPick so it can resume (fine-tune) from a BeakPick run.
from microduck_tidy.pick_profile import (  # noqa: E402
    WIDE_NOISE_X, WIDE_NOISE_Y, WIDE_YAW_RANGE, WIDE_OFFSET_CLIP)

_WIDE = dict(noise_x=WIDE_NOISE_X, noise_y=WIDE_NOISE_Y, yaw_range=WIDE_YAW_RANGE,
             offset_clip=WIDE_OFFSET_CLIP)
register_mjlab_task(
    task_id="Mjlab-BeakPickWide-Flat-MicroDuck",
    env_cfg=make_beak_pick_env_cfg(**_WIDE),
    play_env_cfg=make_beak_pick_env_cfg(play=True, **_WIDE),
    rl_cfg=BeakPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Walking while looking at the floor + turning in place (for searching).
from .search_walk_env_cfg import SearchWalkRlCfg, make_search_walk_env_cfg  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-SearchWalk-Flat-MicroDuck",
    env_cfg=make_search_walk_env_cfg(),
    play_env_cfg=make_search_walk_env_cfg(play=True),
    rl_cfg=SearchWalkRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
