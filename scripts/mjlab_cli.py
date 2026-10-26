"""Run an mjlab CLI (train / play / export) with microduck_fetch tasks registered.

usage: python scripts/mjlab_cli.py train  Mjlab-BeakPick-Flat-MicroDuck --env.scene.num-envs 2048
       python scripts/mjlab_cli.py export Mjlab-BeakPick-Flat-MicroDuck --checkpoint-file <model_N.pt>
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

cmd = sys.argv.pop(1)
import microduck_fetch.tasks  # noqa: E402,F401  (registers our tasks)

if cmd == "train":
    from mjlab.scripts.train import main
elif cmd == "play":
    from mjlab.scripts.play import main
elif cmd == "export":
    from mjlab_microduck.export import main
else:
    raise SystemExit(f"unknown command {cmd!r}")
sys.exit(main())
