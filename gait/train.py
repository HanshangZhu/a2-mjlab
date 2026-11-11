"""Train one of this project's gaits with mjlab (in .venv-gait).

    .venv-gait/bin/python gait/train.py UnrealNav-Velocity-Rough-Unitree-A2 --env.scene.num-envs 4096 --log-root data/gait/logs

The arguments are mjlab's own `train`; this only registers the tasks of ``unrealnav_gait`` first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unrealnav_gait  # noqa: E402, F401  (registers the tasks)
from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
    main()
