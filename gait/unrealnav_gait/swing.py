"""How high the feet are lifted, as a command.

The gait is told how high to swing its feet as it is told how fast to walk: one number a
robot, the height of a foot's sole over the ground under it at the top of a swing. A policy
that lifts its feet 4 cm cannot get up a riser of 15 without first walking into it; one that
always lifts them 20 cm does so on a road too. Told the height, it can do either, and
whatever plans its way says which (a step ahead of 15 cm: swing 20).

The idea is Walk-These-Ways' (``go2_gym/envs/rewards/corl_rewards.py``,
``_reward_feet_clearance_cmd_linear``), which makes the swing height one of fifteen commands
and pays for the feet following a triangle of that height in the swing phase of a gait clock.
Two things are different here. There is no clock: mjlab's two terms for the swing are used,
which need none (the highest point of a swing, judged as the foot lands; and the foot's
height while it moves over the ground), with the commanded height in place of their
constant. And the height is over the ground under the foot, from mjlab's terrain sensor:
Walk-These-Ways measures it from the world's zero, which is a height over the ground on
flat ground only, where that gait was trained.

In training the command is drawn evenly over its range, except on a tile of stairs, where
it is not less than that tile's riser and a margin: told to swing its feet 5 cm at a stair
of 15, a robot is told two things that cannot both be done.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactSensor, TerrainHeightSensor

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

FOOT_RADIUS = 0.032  # the terrain sensor is at the foot's centre: its sole is this much lower


class SwingHeightCommand(CommandTerm):
    cfg: SwingHeightCommandCfg

    def __init__(self, cfg: SwingHeightCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.height = torch.zeros(self.num_envs, 1, device=self.device)
        # The step each tile of the terrain asks a robot to get over, by level and kind: its riser for stairs,
        # its highest block for obstacles, nothing for ground without steps.
        terrain = env.scene.terrain
        generator = terrain.cfg.terrain_generator if terrain is not None else None
        self._terrain, self._step = terrain, None
        if generator is not None and terrain.terrain_origins is not None:
            rows = terrain.terrain_origins.shape[0]
            difficulty = torch.arange(rows, device=self.device) / max(rows - 1, 1)
            low, high = generator.difficulty_range
            difficulty = low + (high - low) * difficulty
            columns = []
            for ground in generator.sub_terrains.values():
                span = getattr(ground, "step_height_range", None) or getattr(ground, "obstacle_height_range", None)
                columns.append(torch.zeros(rows, device=self.device) if span is None else span[0] + (span[1] - span[0]) * difficulty)
            self._step = torch.stack(columns, dim=1)  # [level, kind]
        self.metrics["swing_height"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.height

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        low, high = self.cfg.range
        drawn = torch.empty(len(env_ids), device=self.device).uniform_(low, high)
        if self._step is not None and self.cfg.follow_steps:
            step = self._step[self._terrain.terrain_levels[env_ids], self._terrain.terrain_types[env_ids]]
            needed = step + torch.empty_like(step).uniform_(*self.cfg.margin)
            drawn = torch.where(step > 0.0, torch.maximum(drawn, needed).clamp(max=high), drawn)
        self.height[env_ids, 0] = drawn

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        pass

    def _update_metrics(self) -> None:
        self.metrics["swing_height"] = self.height[:, 0]


@dataclass(kw_only=True)
class SwingHeightCommandCfg(CommandTermCfg):
    range: tuple[float, float]
    """The lowest and the highest swing commanded, metres of the sole over the ground."""
    margin: tuple[float, float] = (0.03, 0.08)
    """On a tile with steps the command is at least the tile's step and a margin drawn from this."""
    follow_steps: bool = True
    """Whether the command is raised to the tile's step at all. Without it the command knows nothing of the ground."""

    def build(self, env: ManagerBasedRlEnv) -> SwingHeightCommand:
        return SwingHeightCommand(self, env)


def _moving(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
    twist = env.command_manager.get_command(command_name)
    return ((torch.norm(twist[:, :2], dim=1) + torch.abs(twist[:, 2])) > threshold).float()


class swing_peak:
    """At each landing that ends a real swing: the square of how far the swing's highest point
    was from the commanded height, as a share of it. A foot that chatters as it lands comes
    down several times within a few hundredths of a second; those are not swings and are not
    judged (mjlab's own term judges them, at a height of nothing, which no lifting can mend)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        sensor = env.scene[cfg.params["height_sensor_name"]]
        assert isinstance(sensor, TerrainHeightSensor)
        self.peak = torch.zeros((env.num_envs, sensor.num_frames), device=env.device)
        self.step_dt = env.step_dt

    def __call__(self, env: ManagerBasedRlEnv, sensor_name: str, height_sensor_name: str, swing_name: str, command_name: str,
                 command_threshold: float, min_air_time: float, at_least: bool = False) -> torch.Tensor:
        contact: ContactSensor = env.scene[sensor_name]
        sole = env.scene[height_sensor_name].data.heights - FOOT_RADIUS
        self.peak = torch.where(contact.data.found == 0, torch.maximum(self.peak, sole), self.peak)
        landed = contact.compute_first_contact(dt=self.step_dt)
        swung = landed & (contact.data.last_air_time > min_air_time)
        wanted = env.command_manager.get_command(swing_name)  # [B, 1]
        short = self.peak / wanted - 1.0
        if at_least:  # a swing higher than wanted is no fault: a riser higher than the aim needs one
            short = short.clamp(max=0.0)
        cost = torch.sum(torch.square(short) * swung.float(), dim=1) * _moving(env, command_name, command_threshold)
        swings = torch.clamp(swung.float().sum(), min=1.0)
        env.extras["log"]["Metrics/swing_peak_mean"] = (self.peak * swung.float()).sum() / swings
        env.extras["log"]["Metrics/swing_peak_of_command"] = ((self.peak / wanted) * swung.float()).sum() / swings
        self.peak = torch.where(landed, torch.zeros_like(self.peak), self.peak)
        return cost


def swing_clearance(env: ManagerBasedRlEnv, height_sensor_name: str, swing_name: str, command_name: str, command_threshold: float,
                    asset_cfg, at_least: bool = False) -> torch.Tensor:
    """While a foot moves over the ground: how far its sole is from the commanded height, times how fast it moves.
    It is nothing for a foot that stands, and least for one that goes up, across at that height, and down.
    With ``at_least`` a foot over that height costs nothing either."""
    robot = env.scene[asset_cfg.name]
    sole = env.scene[height_sensor_name].data.heights - FOOT_RADIUS
    speed = torch.norm(robot.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
    wanted = env.command_manager.get_command(swing_name)
    off = (wanted - sole).clamp(min=0.0) if at_least else torch.abs(sole - wanted)
    return torch.sum(off * speed, dim=1) * _moving(env, command_name, command_threshold)


def swing_ramp(env: ManagerBasedRlEnv, env_ids: torch.Tensor, swing_name: str, start: float, step: float, every: int, top: float) -> dict[str, torch.Tensor]:
    """Raise the height every swing is to reach, a step at a time: ``step`` metres more every ``every`` steps of the
    environment, from ``start`` to ``top``. Asked at once for swings eight times as high as it makes, a gait stops
    walking (a foot that does not land is not judged); asked for a quarter more, it lifts its feet."""
    del env_ids
    term = env.command_manager.get_term(swing_name)
    # Counted in steps this training has itself made. The environment's counter cannot be read for that: carried on
    # from a checkpoint it is set to that checkpoint's count, and after the first reset at that, so that a ramp read
    # off it (or off its value at the first call) began at its top. A jump of the counter is therefore not counted.
    seen = getattr(term, "ramp_seen", None)
    if seen is None:
        term.ramp_made = 0
    elif 0 <= env.common_step_counter - seen <= 1000:
        term.ramp_made += env.common_step_counter - seen
    term.ramp_seen = env.common_step_counter
    aim = min(start + step * (term.ramp_made // every), top)
    term.cfg.range = (aim, aim)
    term.height[:] = aim
    return {"aim": torch.tensor(aim)}
