"""Is the A2 this project's physics runs the A2 the gait is trained on? Array by array.

    .venv-gait/bin/python gait/check_plant.py

Compiles the robot as mjlab's task does (``unrealnav_gait``, with the task's simulation
options) and as the simulation does (``unrealnav.sim.a2``), and prints every difference in
bodies, joints, actuators, colliding geoms and options. There must be none, and this fails if
there is one. (The feet's friction is the file's in both; training then draws each robot's
own.) It then prints what ``contact="mjlab"`` would change, which is what the first gait
trained here was trained on and should not have been.
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "gait"), str(REPO / "src")]
from mjlab.entity import Entity  # noqa: E402
from unrealnav_gait.velocity import a2_rough_env_cfg  # noqa: E402
from unrealnav.sim.a2 import a2_mjcf  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--payload', action='store_true', help='compare the nominal backpack training and deployment plants')
args = parser.parse_args()
cfg = a2_rough_env_cfg(deployment_baseline=True, payload=args.payload)
trained = Entity(cfg.scene.entities["robot"]).spec.compile()
cfg.sim.mujoco.apply(trained)
failed = False
for contact in ("unitree", "mjlab"):
    ours = mujoco.MjModel.from_xml_string(a2_mjcf(contact=contact, payload=args.payload))
    print(f"\n== contact='{contact}' against the training's compiled robot")
    different = []
    # joints, bodies and actuators by name; geoms in order among those that collide
    def by_name(kind, n_a, n_b):
        a = {getattr(trained, kind)(i).name: i for i in range(n_a)}
        b = {getattr(ours, kind)(i).name: i for i in range(n_b)}
        assert set(a) == set(b), (kind, set(a) ^ set(b))
        return [(a[k], b[k], k) for k in a]
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat"):
        for i, j, name in by_name("body", trained.nbody, ours.nbody):
            if name in ("world", "base_link") and field in ("body_pos",):
                continue
            if not np.allclose(getattr(trained, field)[i], getattr(ours, field)[j], atol=1e-9):
                different.append((field, name, getattr(trained, field)[i], getattr(ours, field)[j]))
    for field in ("jnt_range", "jnt_axis", "jnt_pos", "jnt_actfrcrange", "jnt_actfrclimited", "jnt_limited", "jnt_stiffness", "jnt_solref", "jnt_solimp", "jnt_margin"):
        for i, j, name in by_name("joint", trained.njnt, ours.njnt):
            if not np.allclose(getattr(trained, field)[i], getattr(ours, field)[j], atol=1e-9):
                different.append((field, name, getattr(trained, field)[i], getattr(ours, field)[j]))
    for field in ("dof_armature", "dof_damping", "dof_frictionloss", "dof_solref", "dof_solimp"):
        for i, j, name in by_name("joint", trained.njnt, ours.njnt):
            a, b = trained.jnt_dofadr[i], ours.jnt_dofadr[j]
            if not np.allclose(getattr(trained, field)[a], getattr(ours, field)[b], atol=1e-9):
                different.append((field, name, getattr(trained, field)[a], getattr(ours, field)[b]))
    for field in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_forcelimited", "actuator_ctrllimited", "actuator_gear", "actuator_dyntype", "actuator_gaintype", "actuator_biastype", "actuator_trntype"):
        for i, j, name in by_name("actuator", trained.nu, ours.nu):
            if not np.allclose(getattr(trained, field)[i], getattr(ours, field)[j], atol=1e-6):
                different.append((field, name, getattr(trained, field)[i], getattr(ours, field)[j]))
    ta = [g for g in range(trained.ngeom) if trained.geom_contype[g] or trained.geom_conaffinity[g]]
    ob = [g for g in range(ours.ngeom) if ours.geom_contype[g] or ours.geom_conaffinity[g]]
    assert len(ta) == len(ob), (len(ta), len(ob))
    for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity", "geom_condim", "geom_priority", "geom_friction", "geom_solref", "geom_solimp", "geom_solmix", "geom_margin", "geom_gap"):
        seen = {}
        for g, h in zip(ta, ob):
            assert trained.body(trained.geom_bodyid[g]).name == ours.body(ours.geom_bodyid[h]).name
            if not np.allclose(getattr(trained, field)[g], getattr(ours, field)[h], atol=1e-9):
                key = (field, "foot" if trained.geom(g).name.endswith("foot_collision") else "other", str(getattr(trained, field)[g]), str(getattr(ours, field)[h]))
                seen[key] = seen.get(key, 0) + 1
        different += [(f, f"{n} {kind} geoms", a, b) for (f, kind, a, b), n in seen.items()]
    for field in ("timestep", "integrator", "cone", "impratio", "solver", "iterations", "ls_iterations", "tolerance", "ls_tolerance", "ccd_iterations", "noslip_iterations", "jacobian", "disableflags", "enableflags", "gravity", "o_margin", "density", "viscosity", "wind"):
        a, b = getattr(trained.opt, field), getattr(ours.opt, field)
        if not np.allclose(a, b):
            different.append(("opt." + field, "", a, b))
    print(f"   {trained.nbody} bodies, {trained.njnt} joints, {trained.nu} actuators, {len(ta)} colliding geoms; mass {trained.body_mass.sum():.3f} and {ours.body_mass.sum():.3f} kg")
    if not different:
        print("   nothing differs")
    failed |= contact == "unitree" and bool(different)
    for field, name, a, b in different:
        print(f"   {field:22s} {name:20s} training {a}   ours {b}")
sys.exit(1 if failed else 0)
