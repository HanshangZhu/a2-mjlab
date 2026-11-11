"""CPU check of replay timing and reset isolation; run in .venv-gait."""
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import torch

from unrealnav_gait.planner_commands import PlannerVelocityCommandCfg


with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'trace.npz'
    times = np.arange(1000) * 0.02
    commands = np.column_stack([times / 100, times / 200, -times / 300])
    np.savez(path, time=times, command=commands)
    cfg = PlannerVelocityCommandCfg(
        entity_name='robot', resampling_time_range=(4, 8), trace_path=str(path),
        ranges=PlannerVelocityCommandCfg.Ranges(lin_vel_x=(-1, 1.5), lin_vel_y=(-.6, .6), ang_vel_z=(-1, 1)))
    data = SimpleNamespace(root_link_lin_vel_b=torch.zeros(2, 3), root_link_ang_vel_b=torch.zeros(2, 3))
    env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.02, scene={'robot': SimpleNamespace(data=data)})
    term = cfg.build(env)
    term.replay[:] = True
    term.time_left[:] = 100
    for _ in range(5):
        term.compute(.02)
    np.testing.assert_allclose(term.command, np.repeat(commands[5:6], 2, axis=0), atol=1e-8)
    term.compute(torch.tensor([.02, 0]))
    np.testing.assert_allclose(term.command[0], commands[6], atol=1e-8)
    np.testing.assert_allclose(term.command[1], commands[5], atol=1e-8)
    before = term.command[1].clone()
    term.reset(torch.tensor([0]))
    term.compute(0, torch.tensor([0]))
    torch.testing.assert_close(term.command[1], before)
    saved = term.command.clone()
    term._update_command(torch.tensor([0]))
    torch.testing.assert_close(term.command, saved)
    print('planner replay: sample timing, per-env dt and reset isolation passed')
