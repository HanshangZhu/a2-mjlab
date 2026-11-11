# a2-mjlab

Training for the Unitree A2 in [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp, Apache-2.0):

- **a velocity gait** (`UnrealNav-Velocity-{Rough,Stairs,Flat}-Unitree-A2`): blind, on rough ground and stairs, on
  Unitree's own MuJoCo model with its contact parameters (`gait/unrealnav_gait/velocity.py`, `a2.py`);
- variants of it: with a backpack (`...-Payload-...`), with a history, a swing-height command, a local goal;
- **a navigation policy over the frozen gait** (`UnrealNav-Nav-ETH-Unitree-A2`), a reproduction of Lee, Bjelonic,
  Reske, Wellhausen, Miki, Hutter, *Learning Robust Autonomous Navigation and Locomotion for Wheeled-Legged Robots*,
  Science Robotics 2024 ([arXiv 2405.01792](https://arxiv.org/abs/2405.01792)): a height map in, velocity
  commands out at 10 Hz, trained in physics on worlds made by Wave Function Collapse. What the paper gives, what it
  does not and what we chose: `docs/NOTES-ETH-NAVIGATION.md`.

It is the training side of UnrealNav (a simulator for long-range navigation; not public), which runs these policies.
The A2 model files it needs are copied from there (`src/unrealnav/sim/`).

## Setup

```bash
uv venv .venv-gait --python 3.12 && uv pip install --python .venv-gait/bin/python -r gait/requirements.txt
# Unitree's model of the A2 (not included):
git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_mujoco.git external/unitree_mujoco
git -C external/unitree_mujoco sparse-checkout set unitree_robots/a2
```

The backpack tasks read a URDF of the robot with its backpack from `external/a2_payload`, which is not included.
`UnrealNav-Velocity-Payload-Planner-Unitree-A2` also replays a recorded trace of navigation commands
(`data/gait/planner_commands/stack18.npz`), which is not included either.

## Use

```bash
.venv-gait/bin/python gait/check_plant.py      # the robot trained on is the robot the simulator runs: "nothing differs"
.venv-gait/bin/python gait/train.py UnrealNav-Velocity-Rough-Unitree-A2 --env.scene.num-envs 4096 --log-root data/gait/logs
.venv-gait/bin/python gait/export_policy.py data/gait/logs/a2_velocity/<run> gait/policies/a2_rough.npz
.venv-gait/bin/python gait/train.py UnrealNav-Nav-ETH-Unitree-A2 --env.scene.num-envs 2048 --log-root data/gait/logs
```

`gait/policies/a2_rough.npz` is the gait in use (iteration 5,600), as plain arrays with its interface in `meta`.

## Licence

Apache-2.0 (`LICENSE`).
