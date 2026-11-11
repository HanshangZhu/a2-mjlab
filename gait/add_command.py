"""Give a trained gait one more thing to be told, without unlearning what it does.

    .venv-gait/bin/python gait/add_command.py data/gait/logs/a2_velocity/<run> --checkpoint 7600 --mean 0.15 --deviation 0.064 --as swing_from_7600

A gait trained without the swing-height command cannot be carried on as one that has it:
actor and critic are each one input too narrow (the actor one for every frame it is given).
This writes a checkpoint in which both take the new number, last in their observations as
``unrealnav_gait/velocity.py`` puts it, with **weights of zero on it**: it computes exactly
what the old one did, whatever it is told, and training carried on from it starts as the old
gait and learns to listen.

``--mean`` and ``--deviation`` are what the normalisers are given for the new number. They
do not learn it themselves in any useful time: they average over everything seen since the
first iteration, and one more number seen from iteration 7,600 would take as long again to
settle. For a command drawn evenly from a range they are its middle and its width over the
root of twelve.

The optimizer's running averages are widened likewise. Before anything is written the old
and the new actor are run on random observations and must agree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_history import TERMS, forward  # noqa: E402


def append(state: dict[str, torch.Tensor], columns: int, mean: float, deviation: float) -> dict[str, torch.Tensor]:
    """A network's parameters with ``columns`` more inputs at the end, read with weights of zero."""
    new = dict(state)
    weight = state["mlp.0.weight"]
    new["mlp.0.weight"] = torch.cat([weight, torch.zeros(weight.shape[0], columns)], dim=1)
    for name, value in (("obs_normalizer._mean", mean), ("obs_normalizer._var", deviation**2), ("obs_normalizer._std", deviation)):
        new[name] = torch.cat([state[name], torch.full((1, columns), value)], dim=1)
    return new


def widen_averages(optimizer: dict, shapes: dict[torch.Size, int]) -> int:
    """Adam's two running averages of each widened first layer, given the same new columns of zero."""
    changed = 0
    for entry in optimizer.get("state", {}).values():
        for name, value in list(entry.items()):
            if torch.is_tensor(value) and value.shape in shapes:
                entry[name] = torch.cat([value, torch.zeros(value.shape[0], shapes[value.shape])], dim=1)
                changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="the run directory of the gait to start from")
    parser.add_argument("--checkpoint", type=int, required=True, help="its iteration, saved as model_N.pt")
    parser.add_argument("--mean", type=float, required=True, help="of the new command, for the normalisers")
    parser.add_argument("--deviation", type=float, required=True, help="of the new command, for the normalisers")
    parser.add_argument("--as", dest="name", required=True, help="the name of the run directory to write, beside the old one")
    args = parser.parse_args()
    saved = torch.load(args.run / f"model_{args.checkpoint}.pt", map_location="cpu", weights_only=True)
    actor, critic = saved["actor_state_dict"], saved["critic_state_dict"]
    frames, rest = divmod(actor["mlp.0.weight"].shape[1], sum(TERMS))
    if rest:
        raise SystemExit(f"the actor takes {actor['mlp.0.weight'].shape[1]} numbers, which is not a whole number of frames of {sum(TERMS)}: has it a command more already?")
    new_actor, new_critic = append(actor, frames, args.mean, args.deviation), append(critic, 1, args.mean, args.deviation)

    generator = torch.Generator().manual_seed(0)
    worst = 0.0
    for _ in range(200):
        obs = torch.randn(1, actor["mlp.0.weight"].shape[1], generator=generator) * 2.0
        told = torch.rand(1, frames, generator=generator)  # whatever the new command says
        worst = max(worst, float((forward(new_actor, torch.cat([obs, told], dim=1)) - forward(actor, obs)).abs().max()))
    if worst > 1e-4:
        raise SystemExit(f"the widened actor does not compute what the old one does: off by {worst:g}")

    optimizer = saved.get("optimizer_state_dict") or {}
    changed = widen_averages(optimizer, {actor["mlp.0.weight"].shape: frames, critic["mlp.0.weight"].shape: 1})
    if optimizer and changed != 4:
        raise SystemExit(f"expected the optimizer to hold four averages of the two first layers' shapes, found {changed}")
    out = args.run.parent / args.name
    out.mkdir(exist_ok=True)
    torch.save(dict(saved, actor_state_dict=new_actor, critic_state_dict=new_critic, optimizer_state_dict=optimizer), out / f"model_{args.checkpoint}.pt")
    print(f"{out / f'model_{args.checkpoint}.pt'}: {args.run.name} at {args.checkpoint} with one command more: the actor takes "
          f"{new_actor['mlp.0.weight'].shape[1]} numbers ({frames} frames) for {actor['mlp.0.weight'].shape[1]}, the critic "
          f"{new_critic['mlp.0.weight'].shape[1]} for {critic['mlp.0.weight'].shape[1]}; the actor agrees with the old one to {worst:.1e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
