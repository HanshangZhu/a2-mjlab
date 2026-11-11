"""A term that makes a slow command worth obeying.

mjlab pays for tracking the commanded velocity with exp(-error^2 / 0.25): a robot told to walk
at 0.3 m/s that stands still keeps 0.70 of that reward. Every gait trained here therefore did
less than it was told at low speed, and nothing under 0.2 m/s, which is the navigation stack's
whole range; and when lifting the feet high was made to cost something, the gait told 0.3 m/s
stopped walking altogether and stood on three legs.

This is the same error as a share of what was asked: standing still costs 1 whatever the speed
told, and half a turn not made costs half as much. It is added to mjlab's terms, not put in
their place: their wide bell is what a robot on a stair it cannot yet climb learns from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def command_shortfall(env: ManagerBasedRlEnv, command_name: str, floor: float = 0.2, most: float = 1.5, turn_share: float = 0.5,
                      entity_name: str = "robot") -> torch.Tensor:
    """The velocity's error over the speed told (not under ``floor``, so that a command of nearly nothing does
    not blow it up), and ``turn_share`` of the same for the turn; each at most ``most``; nothing for a robot told to stand."""
    robot = env.scene[entity_name]
    command = env.command_manager.get_command(command_name)
    told = torch.norm(command[:, :2], dim=1)
    walking = torch.norm(command[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=1) / told.clamp(min=floor)
    turn = torch.abs(command[:, 2])
    turning = torch.abs(command[:, 2] - robot.data.root_link_ang_vel_b[:, 2]) / turn.clamp(min=floor)
    return walking.clamp(max=most) * (told > 0.05) + turn_share * turning.clamp(max=most) * (turn > 0.05)


def foot_held_up(env: ManagerBasedRlEnv, sensor_name: str, longest: float = 0.6) -> torch.Tensor:
    """The seconds, beyond ``longest``, that each foot has now been off the ground, summed over the feet.
    A swing is over in half a second. A gait made to lift its feet high and told to walk slowly found that it could
    stand on three legs with the fourth held up: nothing judged a foot that never landed."""
    return torch.sum((env.scene[sensor_name].data.current_air_time - longest).clamp(min=0.0), dim=1)
