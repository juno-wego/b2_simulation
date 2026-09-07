"""Unitree B2 + FAIRINO FR3 velocity environment configurations.

A separate task from `Unitree-B2-*`, not a variant of it.  The two share no
config objects and write to different experiment directories, so tuning one
cannot regress the other.

Everything that differs from the plain B2 config traces to one fact: the robot
is now 105 kg instead of 83.5 kg, and 15 kg of that is a 622 mm arm sitting
0.33 m above the trunk that moves under someone else's control.

  * the payload's mass, COM envelope and reaction wrench are randomized
    (see `payload.py`),
  * the base is held flatter and steadier, because tilt at the trunk is
    amplified into a large COM excursion at the arm and into pointing error at
    the LiDAR the SLAM stack runs on,
  * the command-range ceiling is lower (see `curriculum.py`).

Reward *structure* is untouched.  The plain-B2 rewards were the ones that
actually produced a policy that walks; changing their shape at the same time as
adding 21.5 kg would make a bad result impossible to attribute.
"""

from typing import Literal

from src.assets.robots import get_b2_arm_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity import mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from .curriculum import DEPLOY_RANGES, commands_vel_adaptive
from .payload import (
  ARM_COM_RANGE,
  ARM_MASS_ALPHA,
  ARM_REACTION,
  DECK_MASS_ALPHA,
)

TerrainType = Literal["rough", "obstacles"]

# Unchanged from plain B2: the legs are the same legs.
B2_GAIT_PERIOD = 0.7
B2_FOOT_CLEARANCE = 0.15


def unitree_b2_arm_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree B2+FR3 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_b2_arm_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Includes payload_deck_collision and fr3_arm_collision, so grounding the arm
  # terminates the episode the same way grounding the trunk does.
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 3.0
  cfg.viewer.elevation = -10.0

  cfg.observations["actor"].terms["phase"].params["period"] = B2_GAIT_PERIOD
  cfg.observations["critic"].terms["phase"].params["period"] = B2_GAIT_PERIOD
  cfg.rewards["foot_gait"].params["period"] = B2_GAIT_PERIOD
  cfg.rewards["foot_clearance"].params["target_height"] = B2_FOOT_CLEARANCE

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  ##
  # Keep the trunk flat and steady.
  ##
  # A degree of trunk tilt moves the arm's COM by 0.33 m * sin(1 deg) = 6 mm and
  # swings the LiDAR by the same degree.  Doubling these two penalties buys that
  # back; they are the only reward weights that differ from plain B2.

  cfg.rewards["body_orientation_l2"].weight = -2.0
  cfg.rewards["body_ang_vel"].weight = -0.1

  ##
  # The FR3.
  ##

  cfg.events["arm_com"] = EventTermCfg(
    mode="startup",
    func=dr.body_pos,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("fr3_arm",)),
      "operation": "add",
      "ranges": ARM_COM_RANGE,
    },
  )
  cfg.events["arm_mass"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("fr3_arm",)),
      "alpha_range": ARM_MASS_ALPHA,
    },
  )
  cfg.events["deck_mass"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("payload_deck",)),
      "alpha_range": DECK_MASS_ALPHA,
    },
  )
  # Unlike the COM terms above, this one fires during the episode: the arm keeps
  # moving while the robot walks.
  cfg.events["arm_reaction"] = EventTermCfg(
    mode="step",
    func=envs_mdp.apply_body_impulse,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("fr3_arm",)),
      **ARM_REACTION,
    },
  )

  ##
  # Commands.
  ##

  twist_cfg = cfg.commands["twist"]
  assert isinstance(twist_cfg, UniformVelocityCommandCfg)
  twist_cfg.ranges.lin_vel_x = DEPLOY_RANGES["lin_vel_x"]
  twist_cfg.ranges.lin_vel_y = DEPLOY_RANGES["lin_vel_y"]
  twist_cfg.ranges.ang_vel_z = DEPLOY_RANGES["ang_vel_z"]

  cfg.curriculum.pop("command_vel", None)
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=commands_vel_adaptive,
    params={"command_name": "twist"},
  )

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_b2_arm_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree B2+FR3 flat terrain velocity configuration."""
  cfg = unitree_b2_arm_rough_env_cfg(play=play)

  cfg.sim.njmax = 400
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
