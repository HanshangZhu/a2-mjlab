"""Where a training's robots spend their time: on which kind of ground, how hard, and on stairs,
whether on the steps or beside them.

    .venv-gait/bin/python gait/visitation.py data/gait/logs/a2_velocity/<run> [--checkpoint N] [--num-envs 1024] [--seconds 40]

A policy learns what its robots visit. The terrain of mjlab's velocity task is a grid of
tiles 8 m square, one column for each kind of ground and ten rows of rising difficulty; a
robot is reset on the middle of a tile of its kind, told a velocity that is drawn anew every
few seconds, and wanders. On a tile of stairs the steps are a band round the middle: this
counts how much of all robot-time is spent in that band, and at which riser height, which
is the experience the stairs are learnt from. The same count for a changed task (other
proportions, a smaller landing, more steps) says before any training whether the change
does what it is for.

The robots are driven by the run's own policy, on the task as the package now defines it,
with the difficulty levels drawn evenly over all rows (which is where the curriculum of the
runs so far has settled). Nothing is trained and nothing is written.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unrealnav_gait  # noqa: E402  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="a run directory: its policy drives the robots")
    parser.add_argument("--checkpoint", type=int, default=None, help="an iteration saved as model_N.pt; the newest if not given")
    parser.add_argument("--task", default=unrealnav_gait.ROUGH, help=f"whose ground: {unrealnav_gait.ROUGH} or {unrealnav_gait.STAIRS}")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=40.0, help="simulated; two episodes by default")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    saved = {int(re.fullmatch(r"model_(\d+)", p.stem).group(1)): p for p in args.run.glob("model_*.pt")}
    checkpoint = saved[args.checkpoint if args.checkpoint is not None else max(saved)]

    env_cfg, agent_cfg = load_env_cfg(args.task), load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    generator = env_cfg.scene.terrain.terrain_generator
    kinds = list(generator.sub_terrains)
    base = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(args.task)(env, asdict(agent_cfg), device=args.device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    terrain, robot = base.scene.terrain, base.scene["robot"]
    rows = terrain.terrain_origins.shape[0]
    terrain.terrain_levels[:] = torch.randint(0, rows, (args.num_envs,), device=args.device)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    obs, _ = env.reset()

    # Per kind: robot-steps in all, on its own tile's middle, in the band of steps (stairs only), on its rim, and off its tile.
    steps = int(args.seconds / base.step_dt)
    count = {kind: np.zeros(4) for kind in kinds}
    on_steps_by_level = {kind: np.zeros(rows) for kind in kinds}
    band = {}
    for kind, cfg in generator.sub_terrains.items():
        if hasattr(cfg, "step_width"):
            n = int((generator.size[0] - 2 * cfg.border_width - cfg.platform_width) / (2 * cfg.step_width))
            band[kind] = (cfg.platform_width / 2, cfg.platform_width / 2 + n * cfg.step_width, n)
    with torch.inference_mode():
        for _ in range(steps):
            obs, _, _, _ = env.step(policy(obs))
            away = (robot.data.root_link_pos_w[:, :2] - terrain.env_origins[:, :2]).abs().max(dim=1).values.cpu().numpy()
            kind_of, level = terrain.terrain_types.cpu().numpy(), terrain.terrain_levels.cpu().numpy()
            for k, kind in enumerate(kinds):
                mine = kind_of == k
                inner, outer, _ = band.get(kind, (generator.size[0] / 2, generator.size[0] / 2, 0))
                where = np.digitize(away[mine], [inner, outer, generator.size[0] / 2])  # 0 middle, 1 steps, 2 rim, 3 off the tile
                count[kind] += np.bincount(where, minlength=4)
                on_steps_by_level[kind] += np.bincount(level[mine][where == 1], minlength=rows)

    total = sum(c.sum() for c in count.values())
    print(f"\n{checkpoint.parent.name} {checkpoint.name}; {args.num_envs} robots for {args.seconds:g} s; share of all robot-time")
    print(f"{'kind of ground':24s}{'robots':>8s}{'on its middle':>15s}{'ON THE STEPS':>14s}{'on its rim':>12s}{'off its tile':>14s}")
    for kind in kinds:
        c = count[kind] / total
        steps_cell = f"{c[1]:13.1%}" if kind in band else f"{'':>13s}"
        print(f"{kind:24s}{c.sum():8.1%}{c[0]:15.1%} {steps_cell}{c[2]:12.1%}{c[3]:14.1%}")
    on_steps = sum(count[kind][1] for kind in band)
    print(f"\non the steps of stairs with steps of one length: {on_steps / total:.1%} of all robot-time")
    over10 = over15 = 0.0
    for kind, (inner, outer, n) in band.items():
        cfg = generator.sub_terrains[kind]
        low, high = cfg.step_height_range
        risers = low + (high - low) * np.arange(rows) / max(rows - 1, 1)
        shares = on_steps_by_level[kind] / total
        over10, over15 = over10 + shares[risers > 0.10].sum(), over15 + shares[risers >= 0.15].sum()
        print(f"  {kind}: {n} steps of {cfg.step_width:g} m from {inner:g} to {outer:g} m out: {shares.sum():.2%}, over 10 cm {shares[risers > 0.10].sum():.2%}")
    print(f"on risers over 10 cm: {over10:.1%}; of 15 cm and more: {over15:.1%}")
    base.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
