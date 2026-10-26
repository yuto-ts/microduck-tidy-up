"""BeakPick timing and geometry constants, shared by the GPU training task and the CPU
evaluation runner (kept free of mjlab/torch imports)."""
from .beak import MOUTH_RANGE

GP_PERIOD = 5.0          # s, one pick cycle
DESCENT_END = 0.30       # phase: crouch done
HOLD_END = 0.45          # phase: start standing up
RISE_END = 0.80          # phase: fully up, holding
OPEN_START = 0.08        # phase: beak opens
CLOSE_PHASE = 0.36       # phase: beak shuts
OPEN_ANGLE = MOUTH_RANGE[1]

# Nominal pick spot in the robot's yaw frame (m): in front of the toes (toe tip x~0.034),
# under where the crouched beak reaches.
NOMINAL = (0.070, 0.0)
NOISE_X, NOISE_Y = 0.015, 0.020

# Layout of the 13-D command block at the end of the 61-D observation.
CMD_TWIST = slice(0, 3)   # phase encoding [cos 2pi phi, sin 2pi phi, 0]
CMD_BODY_XY = slice(7, 9)  # prop offset (dx, dy) from NOMINAL, yaw frame, clipped +-5 cm
OFFSET_CLIP = 0.05

# "Wide" variant, for picking after walking up to an object: the walking policy cannot
# stop to within +-15 mm, so the pick has to cover a larger placement window and the
# offset observation a larger range. Fine-tuned from the narrow policy.
WIDE_NOISE_X, WIDE_NOISE_Y = 0.030, 0.035
WIDE_YAW_RANGE = 1.57          # any object orientation (a cube repeats every 90 deg)
WIDE_OFFSET_CLIP = 0.08
