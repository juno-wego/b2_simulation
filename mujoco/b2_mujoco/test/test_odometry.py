"""Exercise the real odometry publisher without starting ROS or simulator threads.

Run with the ROS workspace sourced and a Python environment containing MuJoCo:
    python -m pytest test/test_odometry.py
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from builtin_interfaces.msg import Time as TimeMsg

from b2_mujoco.sim_node import B2MujocoSim


SCENE_FILE = Path(__file__).resolve().parents[1] / "models" / "b2_nav_scene.xml"


@pytest.fixture(scope="module")
def model():
  return mujoco.MjModel.from_xml_path(str(SCENE_FILE))


def publish_odometry(model, *, yaw=0.0, linear_world=(0.0, 0.0, 0.0),
                     angular_body=(0.0, 0.0, 0.0), publish_tf=False):
  data = mujoco.MjData(model)
  base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
  qpos_address = model.jnt_qposadr[joint_id]
  qvel_address = model.jnt_dofadr[joint_id]
  data.qpos[qpos_address:qpos_address + 3] = (3.0, -2.0, 1.0)
  data.qpos[qpos_address + 3:qpos_address + 7] = (
    math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
  )
  # MuJoCo free-joint linear velocity is world-frame; angular velocity is local.
  data.qvel[qvel_address:qvel_address + 3] = linear_world
  data.qvel[qvel_address + 3:qvel_address + 6] = angular_body
  mujoco.mj_forward(model, data)

  messages = []
  transforms = []
  owner = SimpleNamespace(
    model=model,
    data=data,
    _base_bid=base_id,
    _twist=np.zeros(6),
    odom_frame="b2/odom",
    base_frame="b2/base_link",
    odom_pub=SimpleNamespace(publish=messages.append),
    publish_gt_tf=publish_tf,
    tf_broadcaster=SimpleNamespace(sendTransform=transforms.append),
  )
  stamp = TimeMsg(sec=12, nanosec=345000000)
  B2MujocoSim._publish_odom(owner, stamp)
  assert len(messages) == 1
  return messages[0], transforms, data, base_id, stamp


@pytest.mark.parametrize("yaw,linear_world,angular_body", [
  (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
  (0.0, (0.24, 0.0, 0.0), (0.0, 0.0, 0.0)),
  (0.0, (-0.24, 0.0, 0.0), (0.0, 0.0, 0.0)),
  (0.0, (0.0, 0.12, 0.0), (0.0, 0.0, 0.0)),
  (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.2)),
  (0.0, (0.24, -0.1, 0.07), (0.1, -0.08, 0.2)),
  (math.radians(35), (0.24, -0.1, 0.07), (0.1, -0.08, 0.2)),
  (math.pi / 2, (0.0, 0.24, 0.0), (0.0, 0.0, 0.2)),
])
def test_published_twist_is_at_base_origin_in_child_frame(
    model, yaw, linear_world, angular_body):
  message, transforms, data, base_id, stamp = publish_odometry(
    model, yaw=yaw, linear_world=linear_world, angular_body=angular_body
  )
  assert transforms == []
  assert message.header.stamp == stamp
  assert message.header.frame_id == "b2/odom"
  assert message.child_frame_id == "b2/base_link"

  position = message.pose.pose.position
  orientation = message.pose.pose.orientation
  np.testing.assert_allclose(
    (position.x, position.y, position.z), data.xpos[base_id], atol=1e-12
  )
  np.testing.assert_allclose(
    (orientation.w, orientation.x, orientation.y, orientation.z),
    data.xquat[base_id], atol=1e-12,
  )

  linear = message.twist.twist.linear
  angular = message.twist.twist.angular
  expected_linear = data.xmat[base_id].reshape(3, 3).T @ np.asarray(linear_world)
  np.testing.assert_allclose(
    (linear.x, linear.y, linear.z), expected_linear, atol=1e-12
  )
  np.testing.assert_allclose(
    (angular.x, angular.y, angular.z), angular_body, atol=1e-12
  )


def test_rotation_does_not_publish_inertial_center_translation(model):
  message, _, data, base_id, _ = publish_odometry(
    model, yaw=math.radians(35), angular_body=(0.1, -0.08, 0.2)
  )
  # The real B2 model has an offset and rotated inertial frame. Its center moves
  # during rotation, while the base_link origin in this case remains stationary.
  assert np.linalg.norm(model.body_ipos[base_id]) > 0.01
  inertial_velocity = np.zeros(6)
  mujoco.mj_objectVelocity(
    model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, inertial_velocity, 0
  )
  assert np.linalg.norm(inertial_velocity[3:]) > 0.001
  linear = message.twist.twist.linear
  np.testing.assert_allclose((linear.x, linear.y, linear.z), 0.0, atol=1e-12)


def test_optional_ground_truth_transform_matches_odometry_pose(model):
  message, transforms, _, _, stamp = publish_odometry(
    model, yaw=math.radians(35), publish_tf=True
  )
  assert len(transforms) == 1
  transform = transforms[0]
  assert transform.header.stamp == stamp
  assert transform.header.frame_id == message.header.frame_id
  assert transform.child_frame_id == message.child_frame_id
  translation = transform.transform.translation
  position = message.pose.pose.position
  np.testing.assert_allclose(
    (translation.x, translation.y, translation.z),
    (position.x, position.y, position.z), atol=1e-12,
  )
  assert transform.transform.rotation == message.pose.pose.orientation
