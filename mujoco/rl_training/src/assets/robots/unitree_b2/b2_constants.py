"""Unitree B2 constants.

The MJCF is generated from Unitree's official `unitree_mujoco` B2 model so that
training physics match the deployment simulator exactly.  Actuator limits and the
nominal pose come from Unitree's own B2 deployment configs:
`unitree_rl_lab/source/.../assets/robots/unitree.py` (M107-24-2 gearbox specs) and
`unitree_rl_lab/deploy/robots/b2/config/config.yaml` (the FixStand target pose that
hands off to the RL policy).
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from src.assets.robots._asset_utils import update_assets
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

B2_XML: Path = SRC_PATH / "assets" / "robots" / "unitree_b2" / "xmls" / "b2.xml"
assert B2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, B2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(B2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##
# B2 drives hip/thigh and calf with the same M107 motor family but different
# gearing: 200 Nm on hip/thigh, 320 Nm on calf.  Stiffness/damping match the
# gains Unitree ships for B2 velocity deployment.

B2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*hip_.*",),
  stiffness=160.0,
  damping=5.0,
  effort_limit=200.0,
  armature=0.1,
)
B2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*thigh_.*",),
  stiffness=160.0,
  damping=5.0,
  effort_limit=200.0,
  armature=0.1,
)
B2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*calf_.*",),
  stiffness=160.0,
  damping=5.0,
  effort_limit=320.0,
  armature=0.1,
)

##
# Keyframes.
##
# Nominal pose == the pose B2's FixStand FSM holds just before it hands control to
# the policy, so a zero action on the first policy step is a no-op on hardware.

B2_DEFAULT_JOINT_POS = {
  ".*_hip_joint": 0.0,
  ".*_thigh_joint": 0.8,
  ".*_calf_joint": -1.5,
}

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.545),
  joint_pos=B2_DEFAULT_JOINT_POS,
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
# Foot collisions are given custom condim, friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

B2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    B2_ACTUATOR_HIP,
    B2_ACTUATOR_THIGH,
    B2_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_b2_robot_cfg() -> EntityCfg:
  """Get a fresh B2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=B2_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_b2_robot_cfg())

  viewer.launch(robot.spec.compile())
