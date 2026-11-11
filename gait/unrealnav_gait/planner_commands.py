"""Mix recorded session command windows, low-speed stop/go, and broad commands."""
from dataclasses import dataclass

import numpy as np
import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg


class PlannerVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        with np.load(cfg.trace_path, allow_pickle=False) as trace:
            times, commands = trace['time'], trace['command']
        if (commands.shape != (len(times), 3) or len(times) < 602
                or not np.isfinite(commands).all() or not np.isfinite(times).all()):
            raise ValueError('planner trace requires at least 602 finite time/command samples')
        self.trace_dt = float(np.median(np.diff(times)))
        if self.trace_dt <= 0 or not np.allclose(np.diff(times), self.trace_dt, atol=1e-6):
            raise ValueError('planner trace must have uniform increasing sample times')
        self.window_steps = int(np.ceil(12 / self.trace_dt)) + 1
        if self.window_steps >= len(times):
            raise ValueError('planner trace must exceed 12 seconds')
        self.trace = torch.as_tensor(commands, dtype=torch.float32, device=self.device)
        self.replay = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cursor = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.elapsed = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        choice = torch.rand(len(env_ids), device=self.device)
        self.replay[env_ids] = choice < 0.5
        replay_ids = env_ids[choice < 0.5]
        self.cursor[replay_ids] = torch.randint(len(self.trace)-self.window_steps, (len(replay_ids),), device=self.device)
        self.elapsed[replay_ids] = 0
        self.time_left[replay_ids] = torch.empty(len(replay_ids), device=self.device).uniform_(4, 12)
        self.is_standing_env[replay_ids] = False
        # 30% synthetic low-speed commands, including stops; 20% retain the
        # original broad ranges and long holding periods to check forgetting.
        slow = env_ids[(choice >= 0.5) & (choice < 0.8)]
        self.vel_command_b[slow, 0] = torch.empty(len(slow), device=self.device).uniform_(-0.35, 0.45)
        self.vel_command_b[slow, 1] = torch.empty(len(slow), device=self.device).uniform_(-0.15, 0.15)
        self.vel_command_b[slow, 2] = torch.empty(len(slow), device=self.device).uniform_(-0.8, 0.8)
        self.is_standing_env[slow] = torch.rand(len(slow), device=self.device) < 0.25
        self.time_left[slow] = torch.empty(len(slow), device=self.device).uniform_(0.06, 3.6)

    def compute(self, dt, env_ids=None):
        if env_ids is None:
            self.elapsed += dt
        else:
            self.elapsed[env_ids] += dt
        super().compute(dt, env_ids)

    def _update_command(self, env_ids=None):
        super()._update_command(env_ids)
        # A pure refresh: reset calls may refresh twice and must not advance time.
        ids = self.replay.nonzero().flatten()
        index = self.cursor[ids] + torch.floor(self.elapsed[ids] / self.trace_dt + 1e-5).long()
        self.vel_command_b[ids] = self.trace[index.clamp(max=len(self.trace)-1)]


@dataclass(kw_only=True)
class PlannerVelocityCommandCfg(UniformVelocityCommandCfg):
    trace_path: str

    def build(self, env):
        return PlannerVelocityCommand(self, env)
