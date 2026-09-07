"""Unitree B2 velocity environment configurations.

Structure mirrors the Go2 config.  B2 is roughly 3x heavier with a 0.35 m shank
(vs Go2's 0.213 m), so the gait period, swing clearance and viewer distance are
scaled up; everything else is shared with the tuned Go2 setup.
"""

from typing import Literal

from src.assets.robots import (
  get_b2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity import mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from .curriculum import DEPLOY_RANGES, commands_vel_adaptive

TerrainType = Literal["rough", "obstacles"]

# Froude-scaled from Go2's 0.6 s: B2's legs are ~1.65x longer, so it trots slower
# and swings its feet higher.
B2_GAIT_PERIOD = 0.7
B2_FOOT_CLEARANCE = 0.15


def unitree_b2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree B2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_b2_robot_cfg()}

  # Set raycast sensor frame to B2 base_link.
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
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
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

  # B2 swings a longer leg: stretch the gait period everywhere it appears.
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

  # Start from the command range Unitree actually deploys their velocity
  # policies with, and let the performance-gated curriculum grow it from there.
  # mjlab's stock schedule doubles the forward speed at a fixed iteration whether
  # or not the policy can follow it, which is what previously drove
  # error_vel_xy from 0.43 to 0.75 and kept it there.
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

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
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


def unitree_b2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree B2 flat terrain velocity configuration."""
  cfg = unitree_b2_rough_env_cfg(play=play)

  # B2's calf carries four collision boxes per leg (Go2 has two), so leave more
  # constraint headroom than the Go2 flat config.
  cfg.sim.njmax = 400
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
