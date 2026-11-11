"""Explicit 5600-compatible payload fine-tuning tasks, independent of swing experiments."""
from __future__ import annotations

import math
from dataclasses import fields, replace
from pathlib import Path

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .tracking import command_shortfall
from .velocity import a2_rough_env_cfg, a2_ppo_runner_cfg
from .planner_commands import PlannerVelocityCommandCfg


def payload_env_cfg(play: bool = False, *, planner_commands: bool = False, shortfall: bool = False,
                    actor_history: int = 1):
    if actor_history < 1:
        raise ValueError('actor_history must be at least one frame')
    cfg = a2_rough_env_cfg(play, deployment_baseline=True, payload=True)
    actor = cfg.observations['actor']
    actor.terms = {name: replace(term, history_length=actor_history)
                   for name, term in actor.terms.items()}
    if not play:
        # Density perturbations scale mass and inertia together, with a physically
        # consistent COM shift. These initial ranges are experimental, not measured.
        cfg.events['payload_inertia'] = EventTermCfg(
            func=envs_mdp.dr.pseudo_inertia, mode='startup',
            params={'asset_cfg': SceneEntityCfg('robot', body_names=('payload',)),
                    'alpha_range': (0.5 * math.log(0.8), 0.5 * math.log(1.2)),
                    't_range': (-0.02, 0.02)})
    if planner_commands:
        twist = cfg.commands['twist']
        twist.heading_command = False
        twist.ranges.heading = None
        twist.rel_heading_envs = twist.rel_forward_envs = 0.0
        twist.rel_world_envs = 0.0
        cfg.commands['twist'] = PlannerVelocityCommandCfg(
            **{f.name: getattr(twist, f.name) for f in fields(twist)},
            trace_path=str(Path(__file__).resolve().parents[2] / 'data/gait/planner_commands/stack18.npz'))
    if shortfall:
        cfg.rewards['command_shortfall'] = RewardTermCfg(
            func=command_shortfall, weight=-2.5, params={'command_name': 'twist'})
    return cfg


class PayloadFineTuneRunner(VelocityOnPolicyRunner):
    """Keep restored Adam moments, but use the fine-tuning task's learning rate."""

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        rate = self.alg.learning_rate
        infos = super().load(path, load_cfg, strict, map_location)
        self.alg.learning_rate = rate
        for group in self.alg.optimizer.param_groups:
            group['lr'] = rate
        print(f'[INFO] Fine-tuning learning rate: {rate}')
        return infos


def payload_runner_cfg():
    cfg = a2_ppo_runner_cfg()
    cfg.experiment_name = 'a2_payload'
    cfg.algorithm.learning_rate = 1e-4
    cfg.algorithm.schedule = 'fixed'
    cfg.save_interval = 100
    cfg.max_iterations = 1000
    return cfg
