"""Blind local-target fine tuning, separate from deployed velocity tasks.

The actor optionally sees target/deadline features. Both controls have identical
commands, terminal arrival rewards, payload DR, and privileged critic inputs.
This is a small goal-conditioned experiment, not a reproduction of perceptive
Hutter-group policies or a replacement for a planner's collision checks.
"""
from dataclasses import dataclass, fields

import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

from .payload import payload_env_cfg


def command_and_features(position, yaw, target, target_yaw, speed, remaining, active):
    """Torch equivalent of policies.local_goal.command_and_features (version 1)."""
    delta = target - position
    c, s = torch.cos(yaw), torch.sin(yaw)
    relative = torch.stack([c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]], dim=1)
    distance = relative.norm(dim=1)
    error = torch.atan2(torch.sin(target_yaw-yaw), torch.cos(target_yaw-yaw))
    valid = active & (remaining > 0) & (speed > 0)
    linear = relative * (torch.minimum(speed, .8*distance) / distance.clamp(min=1e-9))[:, None]
    linear[distance < .08] = 0
    command = torch.cat([linear, (1.5*error).clamp(-1., 1.)[:, None]], dim=1)
    features = torch.cat([relative.clamp(-3., 3.), torch.sin(error)[:, None], torch.cos(error)[:, None],
                          speed[:, None], (remaining/10.).clamp(0., 2.)[:, None], valid[:, None]], dim=1)
    command[~valid] = 0
    features[~valid] = 0
    return command, features


class LocalGoalCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.target = torch.zeros(self.num_envs, 2, device=self.device)
        self.target_yaw = torch.zeros(self.num_envs, device=self.device)
        self.preferred_speed = torch.zeros(self.num_envs, device=self.device)
        self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics['goal_distance'] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        n = len(env_ids)
        def uniform(low, high):
            return torch.empty(n, device=self.device).uniform_(low, high)
        heading = self.robot.data.heading_w[env_ids]
        distance = uniform(.4, 1.5)
        bearing = uniform(-.65, .65)
        backward = torch.rand(n, device=self.device) < .2
        bearing[backward] += torch.pi
        absolute = heading + bearing
        self.target[env_ids] = self.robot.data.root_link_pos_w[env_ids, :2] + distance[:, None]*torch.stack([torch.cos(absolute), torch.sin(absolute)], dim=1)
        # A reverse target still allows facing forward; the gait need not turn pi.
        self.target_yaw[env_ids] = heading + torch.atan2(torch.sin(bearing - backward*torch.pi), torch.cos(bearing - backward*torch.pi)) + uniform(-.25, .25)
        self.preferred_speed[env_ids] = uniform(.15, .4)
        self.active[env_ids] = torch.rand(n, device=self.device) >= .2
        self.time_left[env_ids] = distance/self.preferred_speed[env_ids] + uniform(1.5, 3.)
        stopped = env_ids[~self.active[env_ids]]
        self.time_left[stopped] = torch.empty(len(stopped), device=self.device).uniform_(2., 5.)
        self.is_standing_env[env_ids] = ~self.active[env_ids]

    def current(self):
        return command_and_features(self.robot.data.root_link_pos_w[:, :2], self.robot.data.heading_w,
                                    self.target, self.target_yaw, self.preferred_speed, self.time_left, self.active)

    def _update_command(self, env_ids=None):
        # Pure refresh; resets and observations do not advance any goal timers.
        self.vel_command_b[:] = self.current()[0]

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics['goal_distance'][:] = (self.target-self.robot.data.root_link_pos_w[:, :2]).norm(dim=1)


@dataclass(kw_only=True)
class LocalGoalCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return LocalGoalCommand(self, env)


def goal_observation(env):
    return env.command_manager.get_term('twist').current()[1]


def deadline_arrival(env):
    term = env.command_manager.get_term('twist')
    robot = term.robot
    distance = (term.target-robot.data.root_link_pos_w[:, :2]).norm(dim=1)
    angle = term.target_yaw-robot.data.heading_w
    angle = torch.atan2(torch.sin(angle), torch.cos(angle))
    speed = robot.data.root_link_lin_vel_b[:, :2].norm(dim=1)
    turn = robot.data.root_link_ang_vel_b[:, 2]
    score = torch.exp(-(distance/.2)**2-(angle/.3)**2-(speed/.15)**2-(turn/.3)**2)
    return score * term.active * (term.time_left <= 1.5)


def goal_progress(env):
    term = env.command_manager.get_term('twist')
    delta = term.target-term.robot.data.root_link_pos_w[:, :2]
    distance = delta.norm(dim=1)
    projected = (delta*term.robot.data.root_link_lin_vel_w[:, :2]).sum(dim=1)/distance.clamp(min=.1)
    return projected.clamp(-1., 1.) * term.active * (distance > .2)


def inactive_motion(env):
    term = env.command_manager.get_term('twist')
    return (term.robot.data.root_link_lin_vel_b[:, :2].square().sum(dim=1)
            + .25*term.robot.data.root_link_ang_vel_b[:, 2].square()) * ~term.active


def goal_env_cfg(play=False, *, observe_goal=True):
    cfg = payload_env_cfg(play=play)
    twist = cfg.commands['twist']
    twist.heading_command = False
    twist.ranges.heading = None
    twist.rel_heading_envs = twist.rel_forward_envs = twist.rel_world_envs = 0.
    cfg.commands['twist'] = LocalGoalCommandCfg(**{f.name:getattr(twist, f.name) for f in fields(twist)})
    if observe_goal:
        cfg.observations['actor'].terms['local_goal'] = ObservationTermCfg(func=goal_observation, history_length=1)
    cfg.observations['critic'].terms['local_goal'] = ObservationTermCfg(func=goal_observation)
    # Tracking remains a preference. Arrival in the final 1.5 seconds is primary.
    cfg.rewards['track_linear_velocity'].weight = .2
    cfg.rewards['track_angular_velocity'].weight = .2
    cfg.rewards['deadline_arrival'] = RewardTermCfg(func=deadline_arrival, weight=12.)
    cfg.rewards['goal_progress'] = RewardTermCfg(func=goal_progress, weight=1.)
    cfg.rewards['inactive_motion'] = RewardTermCfg(func=inactive_motion, weight=-2.)
    return cfg
