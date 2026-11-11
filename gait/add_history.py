"""Give a trained gait a memory without unlearning what it does.

    .venv-gait/bin/python gait/add_history.py data/gait/logs/a2_velocity/<run> --checkpoint 6600 --frames 5 --as stairs_memory
    .venv-gait/bin/python gait/train.py <task> ... --agent.resume True --agent.load-run stairs_memory --agent.load-checkpoint model_6600.pt

A policy trained on the present frame alone cannot be carried on as one that is given the
last N frames: its first layer has the wrong width. This writes a checkpoint whose actor
takes N frames and computes **exactly what the old one did**: the new columns of the first
layer that read the newest frame are the old layer, those that read older frames are zero,
and the normaliser's statistics are repeated for every frame. Training carried on from it
(with ``ACTOR_HISTORY`` set to N in ``unrealnav_gait/velocity.py``) starts as the old gait
and can learn to use what it now sees: that a foot met a riser two steps ago.

The layout is mjlab's: each term of the observation in turn, its N frames oldest first.
The critic is copied as it is, and the optimizer's running averages for the first layer are
spread the same way (left at their old width, the first step of training fails). Before
anything is written the two actors are run on random observations with random pasts and
must agree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

TERMS = (3, 3, 3, 12, 12, 12, 3)  # base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, command
ADDED_TO_DEVIATION = 0.01  # what the normaliser adds to the deviation it divides by (rsl_rl's; export_policy.py reads it off a run)


def columns(frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """For each column of the widened input: is it the newest frame of its term, and which old column it stands for."""
    newest, every, begin = [], [], 0
    for size in TERMS:
        for frame in range(frames):
            newest += [frame == frames - 1] * size
            every += list(range(begin, begin + size))
        begin += size
    return torch.tensor(newest), torch.tensor(every)


def spread(old: torch.Tensor, frames: int) -> torch.Tensor:
    """A first layer's weights (or anything of their shape) over ``frames`` frames: as they were on the newest, zero on the rest."""
    newest, every = columns(frames)
    new = old[:, every].clone()
    new[:, ~newest] = 0.0
    return new


def widen(actor: dict[str, torch.Tensor], frames: int) -> dict[str, torch.Tensor]:
    """The actor's parameters for ``frames`` frames of observation in mjlab's layout."""
    if actor["mlp.0.weight"].shape[1] != sum(TERMS):
        raise ValueError(f"the actor takes {actor['mlp.0.weight'].shape[1]} numbers, not one frame of {sum(TERMS)}")
    new = dict(actor)
    new["mlp.0.weight"] = spread(actor["mlp.0.weight"], frames)
    every = columns(frames)[1]
    for name in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
        new[name] = actor[name][:, every].clone()
    return new


def widen_optimizer(state: dict, shape: torch.Size, frames: int) -> tuple[dict, int]:
    """Adam keeps two running averages for every parameter, of the parameter's shape: those of
    the first layer are spread like its weights. Returns the state and how many were changed."""
    changed = 0
    for entry in state.get("state", {}).values():
        for name, value in list(entry.items()):
            if torch.is_tensor(value) and value.shape == shape:
                entry[name] = spread(value, frames)
                changed += 1
    return state, changed


def forward(actor: dict[str, torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
    x = (obs - actor["obs_normalizer._mean"]) / (actor["obs_normalizer._std"] + ADDED_TO_DEVIATION)
    layers = sorted(int(k.split(".")[1]) for k in actor if k.startswith("mlp.") and k.endswith(".weight"))
    for k in layers:
        x = x @ actor[f"mlp.{k}.weight"].T + actor[f"mlp.{k}.bias"]
        if k != layers[-1]:
            x = torch.nn.functional.elu(x)
    return x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="the run directory of the gait to start from")
    parser.add_argument("--checkpoint", type=int, required=True, help="its iteration, saved as model_N.pt")
    parser.add_argument("--frames", type=int, default=5, help="frames of observation the new actor takes")
    parser.add_argument("--as", dest="name", required=True, help="the name of the run directory to write, beside the old one")
    args = parser.parse_args()
    saved = torch.load(args.run / f"model_{args.checkpoint}.pt", map_location="cpu", weights_only=True)
    old = saved["actor_state_dict"]
    new = widen(old, args.frames)

    # The same answer whatever the past holds: the newest frame is what the old actor saw.
    generator = torch.Generator().manual_seed(0)
    worst = 0.0
    for _ in range(200):
        now = torch.randn(1, sum(TERMS), generator=generator) * 2.0
        parts, begin = [], 0
        for size in TERMS:
            parts += [torch.randn(1, size * (args.frames - 1), generator=generator) * 5.0, now[:, begin:begin + size]]
            begin += size
        worst = max(worst, float((forward(new, torch.cat(parts, dim=1)) - forward(old, now)).abs().max()))
    if worst > 1e-4:  # single precision: the same sum over five times as many terms, four fifths of them zero, to a rounding
        raise SystemExit(f"the widened actor does not compute what the old one does: off by {worst:g}")

    optimizer, changed = widen_optimizer(saved.get("optimizer_state_dict") or {}, old["mlp.0.weight"].shape, args.frames)
    if saved.get("optimizer_state_dict") and changed != 2:
        raise SystemExit(f"expected the optimizer to hold two averages of the first layer's shape, found {changed}")

    out = args.run.parent / args.name
    out.mkdir(exist_ok=True)
    torch.save(dict(saved, actor_state_dict=new, optimizer_state_dict=optimizer), out / f"model_{args.checkpoint}.pt")
    print(f"{out / f'model_{args.checkpoint}.pt'}: the actor of {args.run.name} at {args.checkpoint}, taking {args.frames} frames "
          f"({new['mlp.0.weight'].shape[1]} numbers for {old['mlp.0.weight'].shape[1]}); agrees with it to {worst:.1e} whatever the past holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
