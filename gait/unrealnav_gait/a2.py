"""The Unitree A2 for mjlab: Unitree's own MuJoCo model, with what a velocity task needs added.

The model is ``unitree_mujoco/unitree_robots/a2/a2.xml`` (cloned under ``external/``), the
file this project's own physics runs (``unrealnav.sim.a2``, where the numbers below come
from). It is used as it is for everything that is the robot's: masses and inertias, joint
ranges, torque limits, joint armature, damping and friction loss, **and the contact
parameters of every collision shape**. Added here, because the file has none of them:

- names for the collision geoms (``FL_foot_collision``, ``FL_thigh_collision1``, ...) and a
  site at each foot, which contact sensors and rewards are told by name;
- an angular-rate and a velocity sensor at the IMU site and the angular momentum of the
  whole robot, under the names mjlab's velocity task reads;
- no actuators and none of the file's own sensors: mjlab makes position actuators from the
  configuration below, and the file's torque motors would be a second set.

``gait/check_plant.py`` compiles this robot and the simulation's and prints every difference;
there must be none.

**No collision overrides.** The first gait trained here (run ``2026-10-09_16-24-09``) had
mjlab's usual ones: only the feet with friction, every other shape frictionless (condim 1).
It climbed steps of 15 cm in training's contact and stalled at 10 cm in Unitree's, where
shanks have friction: it had learnt to slide its shins over edges. A robot's shins do not
slide (UnrealNav's notes, not public).

Not Unitree's: the PD gains and the standing pose (``unrealnav.sim.a2`` says how they are set).
"""

from __future__ import annotations

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from unrealnav.sim.a2 import A2_XML, DAMPING, EFFORT, STANDING, STANDING_HEIGHT, STIFFNESS

LEGS = ("FR", "FL", "RR", "RL")
BASE = "base_link"
FOOT_OFFSET = (0.0, 0.0, -0.275)  # the foot's centre in the calf's frame
FOOT_GEOMS = tuple(f"{leg}_foot_collision" for leg in LEGS)


def get_spec() -> mujoco.MjSpec:
    if not A2_XML.exists():
        raise FileNotFoundError(f"{A2_XML}: clone unitree_mujoco under external/ (README.md, 'Setup')")
    spec = mujoco.MjSpec.from_file(str(A2_XML))
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for body in spec.bodies:
        count = 0
        for geom in body.geoms:
            kind = geom.classname.name
            if kind == "foot":
                geom.name = f"{body.name.split('_')[0]}_foot_collision"
            elif kind == "collision":
                count += 1
                geom.name = f"{body.name}_collision{count}"
        if body.name.endswith("_calf"):
            body.add_site(name=body.name.split("_")[0], pos=FOOT_OFFSET, size=[0.032, 0.032, 0.032], group=5)
    site = mujoco.mjtObj.mjOBJ_SITE
    spec.add_sensor(name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=site, objname="imu")
    spec.add_sensor(name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER, objtype=site, objname="imu")
    spec.add_sensor(name="root_angmom", type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM, objtype=mujoco.mjtObj.mjOBJ_BODY, objname=BASE)
    return spec


A2_HIP_THIGH = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_thigh_joint"), stiffness=STIFFNESS, damping=DAMPING, effort_limit=EFFORT["hip_joint"])
A2_CALF = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_calf_joint",), stiffness=STIFFNESS, damping=DAMPING, effort_limit=EFFORT["calf_joint"])
A2_ARTICULATION = EntityArticulationInfoCfg(actuators=(A2_HIP_THIGH, A2_CALF), soft_joint_pos_limit_factor=0.9)

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, STANDING_HEIGHT + 0.02), joint_pos={f".*_{part}_joint": angle for part, angle in STANDING.items()}, joint_vel={".*": 0.0})

# A quarter of the torque limit, in radians of target: what mjlab's own robots use.
A2_ACTION_SCALE = {name: 0.25 * cfg.effort_limit / cfg.stiffness for cfg in A2_ARTICULATION.actuators for name in cfg.target_names_expr}


def get_payload_spec() -> mujoco.MjSpec:
    from unrealnav.sim.a2_payload import add_payload
    spec = get_spec()
    add_payload(spec)
    return spec


def get_a2_robot_cfg(payload: bool = False) -> EntityCfg:
    return EntityCfg(init_state=INIT_STATE, spec_fn=get_payload_spec if payload else get_spec, articulation=A2_ARTICULATION)
