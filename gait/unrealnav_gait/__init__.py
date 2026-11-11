"""Gaits trained for this project in mjlab. Importing this registers its tasks."""

import sys
from pathlib import Path

# The robot's numbers are the simulation's (unrealnav.sim), which this environment has no install of.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mjlab.tasks.registry import register_mjlab_task  # noqa: E402
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner  # noqa: E402

from .velocity import a2_flat_env_cfg, a2_ppo_runner_cfg, a2_rough_env_cfg, a2_stairs_env_cfg  # noqa: E402

ROUGH = "UnrealNav-Velocity-Rough-Unitree-A2"
STAIRS = "UnrealNav-Velocity-Stairs-Unitree-A2"
FLAT = "UnrealNav-Velocity-Flat-Unitree-A2"

register_mjlab_task(task_id=ROUGH, env_cfg=a2_rough_env_cfg(), play_env_cfg=a2_rough_env_cfg(play=True),
                    rl_cfg=a2_ppo_runner_cfg(), runner_cls=VelocityOnPolicyRunner)
register_mjlab_task(task_id=FLAT, env_cfg=a2_flat_env_cfg(), play_env_cfg=a2_flat_env_cfg(play=True),
                    rl_cfg=a2_ppo_runner_cfg(), runner_cls=VelocityOnPolicyRunner)
register_mjlab_task(task_id=STAIRS, env_cfg=a2_stairs_env_cfg(), play_env_cfg=a2_stairs_env_cfg(play=True),
                    rl_cfg=a2_ppo_runner_cfg(), runner_cls=VelocityOnPolicyRunner)

from .payload import PayloadFineTuneRunner, payload_env_cfg, payload_runner_cfg

register_mjlab_task(task_id='UnrealNav-Velocity-Payload-Unitree-A2', env_cfg=payload_env_cfg(),
                    play_env_cfg=payload_env_cfg(play=True), rl_cfg=payload_runner_cfg(), runner_cls=PayloadFineTuneRunner)
register_mjlab_task(task_id='UnrealNav-Velocity-Payload-Planner-Unitree-A2', env_cfg=payload_env_cfg(planner_commands=True),
                    play_env_cfg=payload_env_cfg(play=True, planner_commands=True),
                    rl_cfg=payload_runner_cfg(), runner_cls=PayloadFineTuneRunner)
register_mjlab_task(task_id='UnrealNav-Velocity-Payload-Shortfall-Unitree-A2', env_cfg=payload_env_cfg(shortfall=True),
                    play_env_cfg=payload_env_cfg(play=True, shortfall=True),
                    rl_cfg=payload_runner_cfg(), runner_cls=PayloadFineTuneRunner)
register_mjlab_task(task_id='UnrealNav-Velocity-Payload-History-Shortfall-Unitree-A2',
                    env_cfg=payload_env_cfg(shortfall=True, actor_history=5),
                    play_env_cfg=payload_env_cfg(play=True, shortfall=True, actor_history=5),
                    rl_cfg=payload_runner_cfg(), runner_cls=PayloadFineTuneRunner)

from .local_goal import goal_env_cfg

for suffix, observe_goal in [('Goal', True), ('Goal-Control', False)]:
    register_mjlab_task(task_id=f'UnrealNav-Payload-{suffix}-Unitree-A2',
                        env_cfg=goal_env_cfg(observe_goal=observe_goal),
                        play_env_cfg=goal_env_cfg(play=True, observe_goal=observe_goal),
                        rl_cfg=payload_runner_cfg(), runner_cls=PayloadFineTuneRunner)

from .nav_eth import nav_eth_env_cfg, nav_eth_runner_cfg

# ETH's navigation policy (Lee et al., Science Robotics 2024) over the frozen gait: NOTES-ETH-NAVIGATION.md
register_mjlab_task(task_id='UnrealNav-Nav-ETH-Unitree-A2', env_cfg=nav_eth_env_cfg(), play_env_cfg=nav_eth_env_cfg(play=True),
                    rl_cfg=nav_eth_runner_cfg(), runner_cls=None)
