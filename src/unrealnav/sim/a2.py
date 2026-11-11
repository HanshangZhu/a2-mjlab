"""The Unitree A2 model the physics runs: Unitree's own MuJoCo description, with position
actuators in place of its torque motors.

The description is ``unitree_robots/a2/a2.xml`` of github.com/unitreerobotics/unitree_mujoco
(at ``1eb6642``): 12 joints, 40.1 kg, thigh and calf 0.275 m each. Everything that is the
robot's is used as that file has it: masses and inertias, joint ranges, torque limits (120 N m
at hip and thigh, 180 at the calf), joint armature 0.03, damping 0.2 (calf 1.0) and friction
loss 1.0 N m (calf 3.0), and the collision shapes. Changed here:

- **position actuators.** The file's motors take torques. The gait this project trains for the
  robot (``gait/``) is trained on MuJoCo's own position actuators, whose damping the
  integrator treats implicitly, so the model carries those, with the gains the policy is
  trained with; ``ctrl`` is then a joint target.
- **the simulation options**, which the file leaves at MuJoCo's defaults: the training's
  (``TRAINED_OPTIONS``).
- the file's sensors are left out, and its base is put at the origin (the file stands it at
  (-1, 1, 0.7) for Unitree's own scene).

**The contact parameters are Unitree's**, as the file has them (``contact="unitree"``): every
collision shape has full friction (condim 6, priority 1), the feet 1.3 sliding with 0.01
torsional and rolling, and MuJoCo's default contact softness. The gait is trained on the same
(``gait/unrealnav_gait/a2.py``), with the feet's friction drawn per robot from 0.3 to 1.5
(sliding), 1e-4 to 2e-2 (torsional) and 1e-5 to 2e-2 (rolling).

``contact="mjlab"`` is what mjlab's own robots use and the first gait trained here had: only
the feet with friction (1.0, 0.005, 0.0005, on a contact twice as stiff, ``solref`` 0.01 s),
every other shape sliding without any (condim 1, priority 0). That gait climbed steps of
15 cm on this and stalled at 10 cm on Unitree's: it slid its shins over the edges. Kept so
that a gait can be asked the same question again.

Unitree's two files do not agree on the feet: this one says 1.3, the copy in ``unitree_ros``
leaves MuJoCo's default 1.0 and has 0.3 commented out. Neither is a measurement, and it
decides what slope can be stood on at all: the feet have priority over the ground, so every
step has their friction, and 1.3 holds on 52 degrees where the Go2's 0.4 (Unitree's file for
that robot) lets go at 22. ``foot_friction`` overrides it for experiments.

Not in the model: a limit on joint speed. Unitree's URDF gives 22 rad/s for hip and thigh and
14.7 for the calf; MuJoCo's actuators give full torque at any speed.

Not Unitree's at all: the PD gains (``policies/a2.py`` says where they come from) and the
standing pose. No published values for the A2 were found.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

A2_XML = Path(__file__).resolve().parents[3] / "external/unitree_mujoco/unitree_robots/a2/a2.xml"

# The file's torque limits, N m, by joint class.
EFFORT = {"hip_joint": 120.0, "thigh_joint": 120.0, "calf_joint": 180.0}
# Gains of the position actuators the gait in gait/ was trained with: a joint of the model's
# armature (0.03 kg m2) as a second-order system at 10 Hz with a damping ratio of 2, which is
# how mjlab sets the gains of its own quadrupeds. Not Unitree's.
STIFFNESS = 0.03 * (2.0 * math.pi * 10.0) ** 2
DAMPING = 2.0 * 2.0 * 0.03 * (2.0 * math.pi * 10.0)
# Hips 0, thighs 0.7, calves -1.4 rad puts each foot under its thigh joint; the base is then this high.
STANDING = {"hip": 0.0, "thigh": 0.7, "calf": -1.4}
STANDING_HEIGHT = 0.453

# What the training simulated with (gait/unrealnav_gait/velocity.py and mjlab's velocity task).
TRAINED_OPTIONS = {"timestep": 0.005, "integrator": mujoco.mjtIntegrator.mjINT_IMPLICITFAST, "cone": mujoco.mjtCone.mjCONE_ELLIPTIC,
                   "impratio": 10.0, "iterations": 10, "ls_iterations": 20, "ccd_iterations": 500}
UNITREE_FOOT_FRICTION = (1.3, 0.01, 0.01)  # the file's <default class="foot">: sliding, torsional, rolling
MJLAB_FOOT = {"friction": (1.0, 5e-3, 5e-4), "condim": 6, "priority": 1}
MJLAB_OTHER = {"condim": 1, "priority": 0}
MJLAB_SOLREF = (0.01, 1.0)


def a2_spec(stiffness: float = STIFFNESS, damping: float = DAMPING, contact: str = "unitree", foot_friction: float | None = None,
            source: Path = A2_XML, payload: bool = False, imu: bool = False) -> mujoco.MjSpec:
    """The A2 as a spec, not yet compiled. See the module for what ``contact`` chooses.
    ``foot_friction`` replaces the feet's sliding friction, for experiments: the feet have
    priority over the ground, so theirs is the friction of every step the robot takes."""
    if contact not in ("unitree", "mjlab"):
        raise ValueError(f"contact is 'unitree' or 'mjlab', not {contact!r}")
    if not source.exists():
        raise FileNotFoundError(f"{source}: clone unitree_mujoco under external/ (README.md, 'Setup')")
    spec = mujoco.MjSpec.from_file(str(source))
    spec.meshdir = str((source.parent / spec.meshdir).resolve())  # so that the model loads from anywhere
    for name, value in TRAINED_OPTIONS.items():
        setattr(spec.option, name, value)
    for thing in list(spec.sensors) + list(spec.actuators):
        spec.delete(thing)
    spec.body("base_link").pos = [0.0, 0.0, STANDING_HEIGHT]
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        actuator = spec.add_actuator(name=joint.name, target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        actuator.set_to_position(kp=stiffness, kv=damping)
        limit = EFFORT[joint.classname.name]
        actuator.forcerange, actuator.forcelimited = [-limit, limit], mujoco.mjtLimited.mjLIMITED_TRUE
    if contact == "mjlab":
        for geom in spec.geoms:
            kind = geom.classname.name
            if kind not in ("foot", "collision"):
                continue
            geom.solref = list(MJLAB_SOLREF)
            for name, value in (MJLAB_FOOT if kind == "foot" else MJLAB_OTHER).items():
                setattr(geom, name, list(value) if isinstance(value, tuple) else value)
    if foot_friction is not None:
        for geom in spec.geoms:
            if geom.classname.name == "foot":
                # A geom that takes its friction from its class has none of its own yet.
                torsional, rolling = MJLAB_FOOT["friction"][1:] if contact == "mjlab" else UNITREE_FOOT_FRICTION[1:]
                geom.friction = [foot_friction, torsional, rolling]
    if imu:
        # The front lidar's own IMU (sensors/a2.py): MuJoCo's accelerometer and gyro at a site there, in that lidar's axes.
        # They change nothing of how the robot moves.
        from unrealnav.sensors.a2 import FRONT_LIDAR_AXES, FRONT_LIDAR_POS, IMU_SITE

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, FRONT_LIDAR_AXES.flatten())
        spec.body("base_link").add_site(name=IMU_SITE, pos=FRONT_LIDAR_POS.tolist(), quat=quat.tolist(), size=[0.01, 0.01, 0.01], group=4)
        for kind, name in ((mujoco.mjtSensor.mjSENS_ACCELEROMETER, "acc"), (mujoco.mjtSensor.mjSENS_GYRO, "gyro")):
            spec.add_sensor(name=f"{IMU_SITE}_{name}", type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=IMU_SITE)
    if payload:
        # The backpack it is to carry (sim/a2_payload.py), which the gait was not trained with.
        from unrealnav.sim.a2_payload import add_payload

        add_payload(spec)
    return spec


def a2_mjcf(stiffness: float = STIFFNESS, damping: float = DAMPING, contact: str = "unitree", foot_friction: float | None = None,
            source: Path = A2_XML, payload: bool = False, imu: bool = False) -> str:
    """The same as an MJCF string, which is what ``World`` takes."""
    spec = a2_spec(stiffness, damping, contact, foot_friction, source, payload, imu)
    spec.compile()
    return spec.to_xml()
