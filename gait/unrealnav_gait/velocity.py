"""A velocity-tracking gait for the Unitree A2, trained in mjlab: blind, on rough ground and stairs.

mjlab's velocity task for its Go1 (``mjlab.tasks.velocity.config.go1``), given the A2
(``unrealnav_gait.a2``) and changed in three ways:

- **the policy is blind.** The actor does not get the terrain scan; the critic keeps it. In
  this project every exteroceptive sensor comes from Unreal, so a gait that is to be a
  stand-in for the robot's own locomotion must walk on what the robot feels of itself:
  the IMU, the joints, its last action and the command.
- **the commands are the navigation stack's**: up to 1.5 m/s forward and 1 backward,
  0.6 sideways, and 1 rad/s of turn from the start (the stack asks for 0.8 at once), a tenth
  of the robots told to stand, with no speed curriculum on top.
- **more of training is spent on stairs, and on higher ones.** In mjlab's task 11% of all
  robot-time is on the steps of a stair and none of it on a riser over 10 cm; here it is 31%,
  two thirds of that over 10 cm (``gait/visitation.py`` counts it, for this task as it stands,
  before anything is trained). That is what the block of constants below does: more of the
  robots on stairs, a small landing and a narrow rim so that a flight has nine steps for
  five, risers of 4 to 18 cm for 0 to 10, and more robots told to walk straight ahead. It is
  what made a gait that knelt on 15 cm into one that climbs them.

The rewards are mjlab's. Three other changes were tried together and taken out again
(``ACTOR_HISTORY``, ``SWING_HEIGHT``, ``SHANK_ON_GROUND``, ``TRUNK_ON_GROUND`` are what is
left of them, at mjlab's values): with a trunk on the ground ending the episode and a shank
on the ground costly, the policy learnt to stop in front of what it might not get over.
UnrealNav's notes (not public) have the account.

What the gait is for: the checkpoint this project walks its Go2 with was trained on flat
ground only, is stranded by a stop on a slope and does not climb a step.
"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import discrete_obstacles, open_stairs, random_stairs

from .a2 import A2_ACTION_SCALE, BASE, FOOT_GEOMS, LEGS, get_a2_robot_cfg
from .swing import SwingHeightCommandCfg, swing_clearance, swing_peak, swing_ramp
from .tracking import command_shortfall, foot_held_up

# What a training is run with. The values here are those of the gait in use; the write-up says what each was tried at.
ACTOR_HISTORY = 5  # frames of observation the actor is given, the present one last, at 50 Hz (0 or 1: the present alone; mjlab's task: 0)
# How high the feet swing: one of three ways.
# - SWING_AIM: every swing is to reach at least this (metres of the sole over the ground), whatever the ground and
#   without the gait being told anything: a high-stepping gait. In a training the aim is raised to it in steps
#   (SWING_RAMP: where it starts, how much a step adds, and how many iterations a step lasts).
# - SWING_COMMANDED: the height is a command drawn from this range, raised to the riser on stairs, and the gait is
#   told it (unrealnav_gait/swing.py). That is the ground's truth given through the command: a teacher, which needs
#   someone to say the height when it is used (a planner, or a student that estimates it). Tried once, not in use.
# - neither: mjlab's fixed aim (SWING_HEIGHT, of the foot's centre) at mjlab's weights, under which the A2 swung
#   at 2 to 5 cm whatever stood in its way.
SWING_AIM: float | None = 0.20
SWING_RAMP = (0.11, 0.03, 200)  # this stage carries on from one that ended at 0.11; from a low-stepping gait start at 0.05
SWING_COMMANDED: tuple[float, float] | None = None
SWING_COMMAND_WEIGHTS = {"foot_clearance": -2.0, "foot_swing_height": -30.0}  # the second is paid at landings only: a swing half as high as wanted costs about 1 a second
SWING_MIN_AIR = 0.08  # seconds in the air for a landing to count as the end of a swing
SWING_HEIGHT = 0.10
SWING_WEIGHTS = {"foot_clearance": -2.0, "foot_swing_height": -0.25}  # mjlab's own
STEPS_PER_ITERATION = 24  # of the environment, in the runner's settings below
# What a second of standing still costs a robot that is told to walk, whatever speed it is told (unrealnav_gait/tracking.py);
# None: mjlab's tracking terms alone, under which 0.3 m/s is hardly worth obeying.
STANDING_WHEN_TOLD_TO_WALK: float | None = -2.5
FOOT_HELD_UP = -2.0  # a second for each second a foot has been in the air beyond 0.6 s (with the high swing only: see tracking.py)

# How much of training is spent on stairs, which is what they are learnt from (gait/visitation.py measures it).
# mjlab's task has a fifth of its robots on stairs going up and a fifth going down, and of their time a
# third is on the steps: 11% of all robot-time, 1% of it on risers of 15 cm and more.
TERRAIN_SHARE = {"flat": 0.10, "pyramid_stairs": 0.25, "pyramid_stairs_inv": 0.35, "hf_pyramid_slope": 0.07,
                 "hf_pyramid_slope_inv": 0.07, "random_rough": 0.08, "wave_terrain": 0.08}  # of the robots; "_inv" is the way up
STAIR_RISERS = (0.04, 0.18)  # metres: the lowest and the highest riser of the stairs' ten levels (mjlab's task: 0 to 0.10)
STAIR_LANDING = 1.6  # metres across the flat middle of a tile of stairs, where a robot is reset (mjlab's task: 3.0)
STAIR_RIM = 0.4  # metres of flat ground round the steps (mjlab's task: 1.0); with the landing it makes 9 steps a flight for 5
RESET_SPREAD = 0.3  # metres either way from a tile's middle at which a robot is reset (mjlab's task: 0.5), to stay on the landing
FORWARD_ONLY = 0.35  # of the robots are told to walk straight ahead and nothing else (mjlab's task: 0.2): across the steps, not along them
SHANK_ON_GROUND = -0.1  # per physics step of a control step with a shank on the ground (mjlab's own; four steps make it -0.4 a second)
TRUNK_ON_GROUND: float | None = -0.1  # the same for the trunk; None ends the episode instead, as a thigh on the ground does

# The stairs task: a training for stairs alone, run on from a gait that walks (a2_stairs_env_cfg). Four fifths of its
# robots are on something with steps, of many kinds, because one kind is learnt as that kind: the gait trained on
# steps 0.3 m long stumbled on the first riser of a flight it met a little differently.
STAIR_TREADS = (0.25, 0.30, 0.36, 0.45, 0.60)  # metres a step is long: a steep house stair to steps a body length apart
STAIRS_TASK_RISERS = (0.08, 0.20)  # metres over the ten levels: what the gait it starts from already walks is left out below
STAIRS_SHARE = {"up": 0.40, "down": 0.20,  # regular flights, split evenly over the treads
                "stairs_uneven": 0.06,  # every step another height
                "stairs_open": 0.03, "stairs_open_inv": 0.03,  # slabs with nothing under them: a foot can go in under a tread
                "kerbs": 0.08,  # single blocks and pits to step onto and off, 0.4 to 1.5 m across
                "flat": 0.04, "hf_pyramid_slope": 0.03, "hf_pyramid_slope_inv": 0.04, "random_rough": 0.05, "wave_terrain": 0.04}  # so as not to forget

THIGH_GEOMS = tuple(f"{leg}_thigh_collision{i}" for leg in LEGS for i in (1, 2, 3))
CALF_GEOMS = tuple(f"{leg}_calf_collision{i}" for leg in LEGS for i in (1, 2, 3, 4))
BASE_GEOMS = tuple(f"{BASE}_collision{i}" for i in (1, 2, 3, 4, 5))


def a2_rough_env_cfg(play: bool = False, *, deployment_baseline: bool = False, payload: bool = False) -> ManagerBasedRlEnvCfg:
    # The deployed 5600 checkpoint predates the history/high-swing experiments.
    # New fine-tuning tasks select its interface explicitly, without mutating globals.
    actor_history = 1 if deployment_baseline else ACTOR_HISTORY
    swing_aim = None if deployment_baseline else SWING_AIM
    swing_commanded = None if deployment_baseline else SWING_COMMANDED
    shortfall = None if deployment_baseline else STANDING_WHEN_TOLD_TO_WALK
    cfg = make_velocity_env_cfg()
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.mujoco.impratio = 10
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 60  # the A2 has 37 collision geoms to the Go1's 20-odd, all with full friction
    cfg.scene.entities = {"robot": get_a2_robot_cfg(payload=payload)}

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg) and isinstance(sensor.frame, ObjRef)
            sensor.frame.name = BASE
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(ObjRef(type="site", name=leg, entity="robot") for leg in LEGS)
            sensor.pattern = RingPatternCfg.single_ring(radius=0.05, num_samples=4)

    def touching_ground(name: str, geoms: tuple[str, ...], **more) -> ContactSensorCfg:
        return ContactSensorCfg(name=name, primary=ContactMatch(mode="geom", pattern=geoms, entity="robot"),
                                secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), num_slots=1, **more)

    feet = touching_ground("feet_ground_contact", FOOT_GEOMS, reduce="netforce", track_air_time=True)
    thighs = touching_ground("thigh_ground_touch", THIGH_GEOMS, reduce="none", history_length=4)
    shanks = touching_ground("shank_ground_touch", CALF_GEOMS, reduce="none", history_length=4)
    trunk = touching_ground("trunk_ground_touch", BASE_GEOMS, reduce="none", history_length=4)
    itself = ContactSensorCfg(
        name="self_collision", primary=ContactMatch(mode="subtree", pattern=BASE, entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern=BASE, entity="robot"), fields=("found", "force"), reduce="none",
        num_slots=1, history_length=4)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet, itself, thighs, shanks, trunk)
    assert cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
    generator = cfg.scene.terrain.terrain_generator
    generator.curriculum = True
    # Copies: the task's own dictionary of grounds is shared with every other task.
    generator.sub_terrains = {name: replace(ground, proportion=TERRAIN_SHARE[name]) for name, ground in generator.sub_terrains.items()}
    for name in ("pyramid_stairs", "pyramid_stairs_inv"):
        generator.sub_terrains[name] = replace(
            generator.sub_terrains[name], step_height_range=STAIR_RISERS, platform_width=STAIR_LANDING, border_width=STAIR_RIM)
    cfg.events["reset_base"].params["pose_range"].update(x=(-RESET_SPREAD, RESET_SPREAD), y=(-RESET_SPREAD, RESET_SPREAD))

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = A2_ACTION_SCALE
    cfg.viewer.body_name = BASE
    cfg.viewer.distance = 2.5
    cfg.viewer.elevation = -10.0

    # Blind: the actor walks by feel, the critic may look. And the actor remembers: its terms are
    # copies, because the task's are shared with the critic, which is to see the present only.
    actor = cfg.observations["actor"]
    actor.terms = {name: replace(term, history_length=actor_history) for name, term in actor.terms.items() if name != "height_scan"}

    # What the navigation stack will ask of it, from the first iteration.
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (-1.0, 1.5)
    twist.ranges.lin_vel_y = (-0.6, 0.6)
    twist.ranges.ang_vel_z = (-1.0, 1.0)
    twist.rel_forward_envs = FORWARD_ONLY
    cfg.curriculum.pop("command_vel", None)

    # The feet's friction is nobody's measurement (Unitree's two files say 1.3 and 1.0, its Go2's 0.4),
    # so each robot draws its own: sliding, torsional and rolling. Unitree's 1.3, 0.01, 0.01 are inside.
    del cfg.events["foot_friction"]
    for name, axis, ranges, how in (("slide", 0, (0.3, 1.5), {}), ("spin", 1, (1e-4, 2e-2), {"distribution": "log_uniform"}),
                                    ("roll", 2, (1e-5, 2e-2), {"distribution": "log_uniform"})):
        cfg.events[f"foot_friction_{name}"] = EventTermCfg(
            mode="startup", func=envs_mdp.dr.geom_friction,
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS), "operation": "abs", "axes": [axis],
                    "ranges": ranges, "shared_random": True, **how})
    cfg.events["base_com"].params["asset_cfg"].body_names = (BASE,)

    hips_thighs, calves = r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*", r".*(FR|FL|RR|RL)_calf_joint.*"
    cfg.rewards["pose"].params["std_standing"] = {hips_thighs: 0.05, calves: 0.1}
    cfg.rewards["pose"].params["std_walking"] = {hips_thighs: 0.3, calves: 0.6}
    cfg.rewards["pose"].params["std_running"] = {hips_thighs: 0.3, calves: 0.6}
    cfg.rewards["upright"].params["asset_cfg"].body_names = (BASE,)
    cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (BASE,)
    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = LEGS
    for name in ("body_ang_vel", "angular_momentum", "air_time"):
        cfg.rewards[name].weight = 0.0
    assert swing_aim is None or swing_commanded is None
    if swing_aim is None and swing_commanded is None:
        for name, weight in SWING_WEIGHTS.items():
            cfg.rewards[name].weight = weight
            cfg.rewards[name].params["target_height"] = SWING_HEIGHT
    else:
        # The two terms for the swing aim at a height that a command term holds: one height for all, raised in steps
        # and not shown to the gait; or one drawn for each robot, which actor and critic are told (last in both).
        told = swing_commanded is not None
        start = swing_commanded if told else (SWING_RAMP[0], SWING_RAMP[0])
        cfg.commands["swing"] = SwingHeightCommandCfg(range=start, follow_steps=told, resampling_time_range=twist.resampling_time_range)
        if told:
            seen = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "swing"})
            actor.terms["swing_height"] = replace(seen, history_length=actor_history)
            cfg.observations["critic"].terms = {**cfg.observations["critic"].terms, "swing_height": seen}
        elif not play:
            cfg.curriculum["swing_aim"] = CurriculumTermCfg(func=swing_ramp, params={
                "swing_name": "swing", "start": SWING_RAMP[0], "step": SWING_RAMP[1], "every": SWING_RAMP[2] * STEPS_PER_ITERATION, "top": swing_aim})
        common = {"height_sensor_name": "foot_height_scan", "swing_name": "swing", "command_name": "twist", "command_threshold": 0.05, "at_least": not told}
        cfg.rewards["foot_swing_height"] = RewardTermCfg(
            func=swing_peak, weight=SWING_COMMAND_WEIGHTS["foot_swing_height"],
            params={"sensor_name": feet.name, "min_air_time": SWING_MIN_AIR, **common})
        cfg.rewards["foot_clearance"] = RewardTermCfg(
            func=swing_clearance, weight=SWING_COMMAND_WEIGHTS["foot_clearance"],
            params={"asset_cfg": SceneEntityCfg("robot", site_names=LEGS), **common})

    if shortfall is not None:
        cfg.rewards["command_shortfall"] = RewardTermCfg(func=command_shortfall, weight=shortfall, params={"command_name": "twist"})
    if swing_aim is not None:
        cfg.rewards["foot_held_up"] = RewardTermCfg(func=foot_held_up, weight=FOOT_HELD_UP, params={"sensor_name": feet.name})
    cfg.rewards["self_collisions"] = RewardTermCfg(func=mdp.self_collision_cost, weight=-0.1, params={"sensor_name": itself.name})
    cfg.rewards["shank_collision"] = RewardTermCfg(func=mdp.self_collision_cost, weight=SHANK_ON_GROUND, params={"sensor_name": shanks.name})

    # A thigh on the ground is the end of the episode; a trunk on the ground costs, or ends it too.
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(func=mdp.illegal_contact, params={"sensor_name": thighs.name})
    if TRUNK_ON_GROUND is None:
        cfg.terminations["trunk_contact"] = TerminationTermCfg(func=mdp.illegal_contact, params={"sensor_name": trunk.name})
    else:
        cfg.rewards["trunk_collision"] = RewardTermCfg(func=mdp.self_collision_cost, weight=TRUNK_ON_GROUND, params={"sensor_name": trunk.name})

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(func=envs_mdp.randomize_terrain, mode="reset", params={})
        generator.curriculum, generator.num_cols, generator.num_rows, generator.border_width = False, 5, 5, 10.0
    return cfg


def a2_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The rough task with its ground exchanged for stairs of many kinds: for training on from a gait that walks."""
    cfg = a2_rough_env_cfg(play=play)
    assert cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
    generator = cfg.scene.terrain.terrain_generator
    rough = generator.sub_terrains
    middle = dict(platform_width=STAIR_LANDING, border_width=STAIR_RIM)
    grounds = {}
    for tread in STAIR_TREADS:
        share = 1.0 / len(STAIR_TREADS)
        grounds[f"stairs_up_{100 * tread:.0f}"] = replace(
            rough["pyramid_stairs_inv"], step_width=tread, step_height_range=STAIRS_TASK_RISERS, proportion=STAIRS_SHARE["up"] * share)
        grounds[f"stairs_down_{100 * tread:.0f}"] = replace(
            rough["pyramid_stairs"], step_width=tread, step_height_range=STAIRS_TASK_RISERS, proportion=STAIRS_SHARE["down"] * share)
    grounds["stairs_uneven"] = random_stairs(proportion=STAIRS_SHARE["stairs_uneven"], step_width=0.4, step_height_range=STAIRS_TASK_RISERS, **middle)
    for name, inverted in (("stairs_open", False), ("stairs_open_inv", True)):
        grounds[name] = open_stairs(proportion=STAIRS_SHARE[name], step_height_range=STAIRS_TASK_RISERS, step_width_range=(0.3, 0.5), inverted=inverted, **middle)
    grounds["kerbs"] = discrete_obstacles(proportion=STAIRS_SHARE["kerbs"], obstacle_height_range=(0.03, 0.18), obstacle_width_range=(0.4, 1.5),
                                          num_obstacles=25, platform_width=STAIR_LANDING, border_width=STAIR_RIM)
    for name in ("flat", "hf_pyramid_slope", "hf_pyramid_slope_inv", "random_rough", "wave_terrain"):
        grounds[name] = replace(rough[name], proportion=STAIRS_SHARE[name])
    generator.sub_terrains = grounds
    return cfg


def a2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """The same on a plane: for a quick check that the robot and the task hold together."""
    cfg = a2_rough_env_cfg(play=play)
    cfg.sim.njmax, cfg.sim.nconmax = 300, None
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    gone = {"terrain_scan", "self_collision", "thigh_ground_touch", "shank_ground_touch", "trunk_ground_touch"}
    cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name not in gone)
    del cfg.observations["critic"].terms["height_scan"]
    cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
    for name in ("self_collisions", "shank_collision", "trunk_collision"):
        cfg.rewards.pop(name, None)
    cfg.terminations.pop("illegal_contact", None)
    cfg.terminations.pop("trunk_contact", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["fell_over"] = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
    cfg.curriculum.pop("terrain_levels", None)
    return cfg


def a2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """mjlab's settings for its Go1, under another name and with a nearer horizon."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5,
            num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
        experiment_name="a2_velocity", save_interval=100, num_steps_per_env=STEPS_PER_ITERATION, max_iterations=6_000,
        logger="tensorboard")  # logs stay on this machine: nothing is sent to a tracking service
