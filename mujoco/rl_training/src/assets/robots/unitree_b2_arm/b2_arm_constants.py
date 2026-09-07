"""Unitree B2 + FAIRINO FR3 constants.

Separate lineage from `unitree_b2`: separate MJCF, separate EntityCfg, separate
task, separate ONNX.  Nothing here is imported by the plain B2 config and nothing
here writes over it.

The quadruped half is identical to `unitree_b2` — same generated geometry, same
M107 actuator limits, same FixStand nominal pose — because the hardware is the
same robot and the same `FixStand -> policy` handoff has to work.  What differs
is the 21.5 kg payload on the back (see `tools/make_b2_arm_asset.py`) and,
consequently, the stand height.

Actuator gains stay at Unitree's deployed kp 160 / kd 5.  Raising them would be
the easy way to stop the legs sagging under the extra 21.5 kg, but those gains
are what Unitree validated on B2 hardware, and the deployment reads them straight
out of this policy's ONNX metadata.  The policy compensates for the sag by
commanding a different joint target instead, which costs nothing on hardware.
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

B2_ARM_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_b2_arm" / "xmls" / "b2_arm.xml"
)
assert B2_ARM_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, B2_ARM_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(B2_ARM_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

B2_ARM_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*hip_.*",),
  stiffness=160.0,
  damping=5.0,
  effort_limit=200.0,
  armature=0.1,
)
B2_ARM_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*thigh_.*",),
  stiffness=160.0,
  damping=5.0,
  effort_limit=200.0,
  armature=0.1,
)
B2_ARM_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*calf_.*",),
  stiffness=160.0,
  damping=5.0,
  effort_limit=320.0,
  armature=0.1,
)

##
# Keyframes.
##
# Same FixStand pose as plain B2 so a zero first action is still a no-op at
# handoff.  Spawn height is 10 mm higher: the extra 21.5 kg sags the legs at
# kp 160, and starting at the plain-B2 0.545 m drops the feet into the ground.

B2_ARM_DEFAULT_JOINT_POS = {
  ".*_hip_joint": 0.0,
  ".*_thigh_joint": 0.8,
  ".*_calf_joint": -1.5,
}

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.555),
  joint_pos=B2_ARM_DEFAULT_JOINT_POS,
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# `payload_deck_collision` and `fr3_arm_collision` are picked up by the
# `.*_collision` pattern, so the payload counts as an illegal-contact body: if
# the arm touches the ground the episode ends, exactly as for the trunk.
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

B2_ARM_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    B2_ARM_ACTUATOR_HIP,
    B2_ARM_ACTUATOR_THIGH,
    B2_ARM_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_b2_arm_robot_cfg() -> EntityCfg:
  """Get a fresh B2+FR3 robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=B2_ARM_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_b2_arm_robot_cfg())

  viewer.launch(robot.spec.compile())
