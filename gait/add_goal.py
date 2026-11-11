"""Prepare matched goal/velocity-only continuations with identical initial actors."""
import argparse
import json
from pathlib import Path

import torch

from add_command import append, widen_averages
from add_history import forward


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--checkpoint', type=int, required=True)
    p.add_argument('--as', dest='name', required=True)
    p.add_argument('--control', action='store_true')
    a = p.parse_args()
    out = a.run.parent/a.name
    if out.exists():
        raise FileExistsError(out)
    saved = torch.load(a.run/f'model_{a.checkpoint}.pt', map_location='cpu', weights_only=True)
    actor, critic = saved['actor_state_dict'], saved['critic_state_dict']
    assert actor['mlp.0.weight'].shape[1] == 48, 'This experiment starts from a single-frame velocity actor'
    new_actor = actor if a.control else append(actor, 7, 0., 1.)
    new_critic = append(critic, 7, 0., 1.)
    generator = torch.Generator().manual_seed(0)
    worst = 0.
    for _ in range(200):
        obs = torch.randn(1, 48, generator=generator)*2
        extra = torch.randn(1, 7, generator=generator)*2
        given = obs if a.control else torch.cat([obs, extra], dim=1)
        worst = max(worst, float((forward(new_actor, given)-forward(actor, obs)).abs().max()))
    assert worst < 1e-4, worst
    shapes = {critic['mlp.0.weight'].shape: 7}
    if not a.control:
        shapes[actor['mlp.0.weight'].shape] = 7
    changed = widen_averages(saved['optimizer_state_dict'], shapes)
    assert changed == (2 if a.control else 4), changed
    out.mkdir()
    torch.save(dict(saved, actor_state_dict=new_actor, critic_state_dict=new_critic), out/f'model_{a.checkpoint}.pt')
    report = dict(source=str(a.run), checkpoint=a.checkpoint, control=a.control,
                  actor_max_error=worst, widened_adam_tensors=changed, goal_layout_version=1)
    (out/'initialization.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
