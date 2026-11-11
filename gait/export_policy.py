"""A policy trained here, written out as plain arrays the simulation runs without torch.

    .venv-gait/bin/python gait/export_policy.py data/gait/logs/a2_velocity/<run> gait/policies/a2_rough.npz
    .venv-gait/bin/python gait/export_policy.py data/gait/logs/a2_velocity/<run> out.npz --checkpoint 2500

mjlab's runner writes an ONNX file beside its checkpoints at every save: the actor with its
observation normaliser, and in the file's metadata everything a deployment must reproduce
(joint order, default pose, PD gains, action scale, what the observation is made of). This
reads that file and writes a ``.npz`` that ``unrealnav.policies.a2`` loads with numpy alone:

    mean, scale        the observation is (obs - mean) / scale
    weight0.., bias0.. the layers; ELU between them, none after the last
    meta               JSON: the ONNX metadata, how many frames of observation the policy takes,
                       the run's command ranges and control step, and which checkpoint this is

The graph must be exactly that (normalise, then linear layers with ELU between): anything
else is refused, not approximated. The arrays are checked against onnx's own reference
evaluation of the file on random observations before anything is written.

The ONNX file is the newest checkpoint only. ``--checkpoint N`` writes an earlier one: the
weights and the normaliser's mean and deviation from ``model_N.pt``, the rest from the ONNX
file. How the two relate (the graph divides by the deviation plus a constant) is not assumed:
it is read off the ONNX file and the checkpoint it was written with, which must then agree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

LISTS = ("joint_names", "command_names", "observation_names")
NUMBERS = ("joint_stiffness", "joint_damping", "default_joint_pos", "action_scale")


def read(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    model = onnx.load(str(path))
    constant = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in model.graph.initializer}
    nodes = list(model.graph.node)
    kinds = [n.op_type for n in nodes]
    layers = (len(kinds) - 1) // 2
    if kinds != ["Sub", "Div"] + ["Gemm", "Elu"] * (layers - 1) + ["Gemm"]:
        raise ValueError(f"{path}: not a normaliser and a plain ELU network, but {kinds}")
    arrays = {"mean": constant[nodes[0].input[1]].reshape(-1), "scale": constant[nodes[1].input[1]].reshape(-1)}
    for k, node in enumerate(n for n in nodes if n.op_type == "Gemm"):
        given = {a.name: (a.f if a.type == onnx.AttributeProto.FLOAT else a.i) for a in node.attribute}
        if given.get("alpha", 1.0) != 1.0 or given.get("beta", 1.0) != 1.0 or given.get("transB", 0) != 1 or given.get("transA", 0):
            raise ValueError(f"{path}: a linear layer written another way: {given}")
        arrays[f"weight{k}"], arrays[f"bias{k}"] = constant[node.input[1]], constant[node.input[2]]
    for node in nodes:
        if node.op_type == "Elu" and any(a.name == "alpha" and a.f != 1.0 for a in node.attribute):
            raise ValueError(f"{path}: an ELU with another alpha")
    said = {p.key: p.value for p in model.metadata_props}
    meta = {key: said[key].split(",") for key in LISTS}
    meta.update({key: [float(v) for v in said[key].split(",")] for key in NUMBERS})
    if any(float(v) != 1.0 for v in said["observation_terms_scale"].split(",")):
        raise ValueError(f"{path}: observation_terms_scale is {said['observation_terms_scale']}: scaled observations are not handled")
    # How many frames of each term the policy is given, oldest first (0 and 1 are both the present alone).
    # They must be the same for every term, and flattened into the observation, which is what the simulation builds.
    frames = {max(int(float(v)), 1) for v in said["observation_terms_history_length"].split(",")}
    if len(frames) != 1 or any(float(v) != 1.0 for v in said.get("observation_terms_flatten_history_dim", "1").split(",")):
        raise ValueError(f"{path}: histories of {said['observation_terms_history_length']} frames, not one length for all terms flattened")
    meta["history_length"] = frames.pop()
    if "local_goal" in meta["observation_names"]:
        meta["goal_layout_version"] = 1
    return arrays, meta


def from_checkpoint(path: Path, like: dict[str, np.ndarray], add_to_deviation: float | None = None) -> tuple[dict[str, np.ndarray], float]:
    """The same arrays out of one of the runner's checkpoints, and the constant the graph adds
    to the deviation: given, or else read off ``like`` (the ONNX file's arrays, of this checkpoint)."""
    import torch  # only here: the plain export needs onnx alone

    actor = torch.load(path, map_location="cpu", weights_only=True)["actor_state_dict"]
    deviation = actor["obs_normalizer._std"].numpy().reshape(-1).astype(np.float32)
    if add_to_deviation is None:
        added = like["scale"] - deviation
        if np.ptp(added) > 1e-6:
            raise ValueError(f"{path}: the graph's divisor is not this checkpoint's deviation plus one constant")
        add_to_deviation = float(added.mean())
    arrays = {"mean": actor["obs_normalizer._mean"].numpy().reshape(-1).astype(np.float32), "scale": deviation + np.float32(add_to_deviation)}
    for k in range(sum(name.startswith("weight") for name in like)):
        arrays[f"weight{k}"], arrays[f"bias{k}"] = actor[f"mlp.{2 * k}.weight"].numpy(), actor[f"mlp.{2 * k}.bias"].numpy()
    if {k: v.shape for k, v in arrays.items()} != {k: v.shape for k, v in like.items()}:
        raise ValueError(f"{path}: not the network of the run's ONNX file")
    return arrays, add_to_deviation


def forward(arrays: dict[str, np.ndarray], obs: np.ndarray) -> np.ndarray:
    """What ``unrealnav.policies.a2`` computes; repeated here so that the check below is of the arrays."""
    x = (obs - arrays["mean"]) / arrays["scale"]
    last = max(int(k[6:]) for k in arrays if k.startswith("weight"))
    for k in range(last + 1):
        x = x @ arrays[f"weight{k}"].T + arrays[f"bias{k}"]
        if k < last:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
    return x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="a run directory of mjlab's (its .onnx, model_*.pt and params/)")
    parser.add_argument("out", type=Path, help="the .npz to write")
    parser.add_argument("--checkpoint", type=int, default=None, help="an iteration saved as model_N.pt, instead of the newest")
    args = parser.parse_args()
    found = sorted(args.run.glob("*.onnx"))
    if len(found) != 1:
        raise SystemExit(f"{args.run}: expected one .onnx, found {len(found)}")
    arrays, meta = read(found[0])

    # Which checkpoint the file is: the runner writes it with each model_N.pt, so the newest one not after it.
    written = found[0].stat().st_mtime
    saved = [(int(re.fullmatch(r"model_(\d+)", p.stem).group(1)), p.stat().st_mtime) for p in args.run.glob("model_*.pt")]
    at = max((n for n, when in saved if when <= written + 5.0), default=None)
    env = yaml.safe_load(re.sub(r"!!python/\S+", "", (args.run / "params" / "env.yaml").read_text()))
    ranges = env["commands"]["twist"]["ranges"]
    if "swing" in env["commands"]:
        meta["swing_range"] = [float(v) for v in env["commands"]["swing"]["range"]]
    meta.update(
        task="mjlab local goal" if "local_goal" in meta["observation_names"] else "mjlab velocity", run=args.run.name, iteration=at,
        control_dt=float(env["decimation"]) * float(env["sim"]["mujoco"]["timestep"]), timestep=float(env["sim"]["mujoco"]["timestep"]),
        command_ranges={"forward": list(ranges["lin_vel_x"]), "sideways": list(ranges["lin_vel_y"]), "yaw": list(ranges["ang_vel_z"])})

    reference = ReferenceEvaluator(onnx.load(str(found[0])))
    worst = 0.0
    for obs in np.random.default_rng(0).normal(0.0, 2.0, size=(200, 1, arrays["mean"].size)).astype(np.float32):
        worst = max(worst, float(np.abs(forward(arrays, obs) - reference.run(None, {"obs": obs})[0]).max()))
    if worst > 1e-4:
        raise SystemExit(f"the arrays do not compute what {found[0].name} does: off by {worst:g}")

    agrees = f"agrees with the ONNX file to {worst:.1e}"
    if args.checkpoint is not None and args.checkpoint != at:
        # The newest checkpoint must give back the ONNX file's arrays exactly; then the wanted one is read the same way.
        again, added = from_checkpoint(args.run / f"model_{at}.pt", arrays)
        if any(np.abs(again[k] - arrays[k]).max() > 1e-6 for k in arrays):  # the divisor is a sum in float32: to a rounding
            raise SystemExit(f"model_{at}.pt does not hold what {found[0].name} computes: an earlier checkpoint cannot be trusted either")
        arrays, _ = from_checkpoint(args.run / f"model_{args.checkpoint}.pt", arrays, added)
        meta["iteration"], at = args.checkpoint, args.checkpoint
        agrees = f"read as model_{meta['iteration']}.pt; the newest checkpoint read the same way is the ONNX file to 1e-6"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, meta=np.array(json.dumps(meta)), **arrays)
    sizes = " -> ".join(str(arrays[f"weight{k}"].shape[1]) for k in range(sum(k.startswith("weight") for k in arrays)))
    print(f"{args.out}: iteration {at} of {args.run.name}; {sizes} -> {len(meta['joint_names'])}; "
          f"every {meta['control_dt'] * 1000:g} ms; {agrees}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
