#!/usr/bin/env python3
"""Velocity-policy runner for the Unitree B2.

Turns `/cmd_vel` into low-level joint commands by running the trained RL velocity
policy at 50 Hz.  Input is `unitree_go/LowState`, output is `unitree_go/LowCmd`,
so this node is transport-identical against the MuJoCo simulator and against the
physical B2 -- only the DDS peer changes.

Startup follows B2's own FSM (`unitree_rl_lab/deploy/robots/b2/config/config.yaml`):
damping, then a scripted fold-and-stand, then hand-off to the policy.  The policy's
nominal pose *is* the stand pose, so the first policy action is a no-op.

Everything the policy needs about the robot -- joint order, nominal pose, PD gains,
action scale, observation layout -- is read from the metadata that mjlab attaches to
the exported ONNX, so the runner cannot drift out of sync with the checkpoint.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path

import numpy as np

# Must precede the imports below: it puts the b2sim virtualenv on sys.path
# when the node was launched without sourcing b2_env.sh.
import b2_mujoco._venv  # noqa: F401
import onnxruntime as ort
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
  QoSDurabilityPolicy,
  QoSHistoryPolicy,
  QoSProfile,
  QoSReliabilityPolicy,
)
from std_msgs.msg import String
from unitree_go.msg import LowCmd, LowState

NUM_MOTORS = 12

# Unitree SDK motor order; the index each joint occupies in LowState/LowCmd.
SDK_JOINT_NAMES = [
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

EXPECTED_OBS = ("base_ang_vel", "projected_gravity", "command", "phase",
                "joint_pos", "joint_vel", "actions")

# B2's stock FixStand script: fold to a crouch by t=1 s, push up to stand by t=3 s.
STAND_KEYFRAMES = ((1.0, (0.0, 1.36, -2.65)), (3.0, (0.0, 0.8, -1.5)))
STAND_KP, STAND_KD = 400.0, 8.0
DAMPING_KD = 10.0


def _quat_to_projected_gravity(q: np.ndarray) -> np.ndarray:
  """Gravity direction in the body frame, from a (w, x, y, z) body orientation.

  Equals R^T @ (0, 0, -1); level attitude gives (0, 0, -1).
  """
  w, x, y, z = q
  return np.array([
    -2.0 * (x * z - w * y),
    -2.0 * (y * z + w * x),
    -(1.0 - 2.0 * (x * x + y * y)),
  ])


class B2PolicyRunner(Node):
  def __init__(self) -> None:
    super().__init__("b2_policy_runner")

    default_policy = str(
      Path(__file__).resolve().parents[1] / "policy" / "b2_velocity.onnx"
    )
    self.declare_parameter("policy_file", default_policy)
    self.declare_parameter("low_state_topic", "/lowstate")
    self.declare_parameter("low_cmd_topic", "/lowcmd")
    self.declare_parameter("cmd_vel_topic", "/cmd_vel")
    self.declare_parameter("control_rate_hz", 50.0)
    self.declare_parameter("gait_period", 0.7)
    # Command clamps: the velocity range the policy was trained over.
    self.declare_parameter("max_lin_vel_x", 1.5)
    self.declare_parameter("min_lin_vel_x", -1.0)
    self.declare_parameter("max_lin_vel_y", 0.8)
    self.declare_parameter("max_ang_vel_z", 1.0)
    self.declare_parameter("cmd_vel_timeout", 0.5)
    self.declare_parameter("auto_start", True)
    self.declare_parameter("damping_duration", 0.5)

    policy_file = self.get_parameter("policy_file").value
    if not Path(policy_file).is_file():
      # b2_velocity.onnx and b2_arm_velocity.onnx come from different tasks and
      # different log directories, so point at the right one.
      arm = "arm" in Path(policy_file).stem
      task = "Unitree-B2Arm-Flat" if arm else "Unitree-B2-Flat"
      experiment = "b2_arm_velocity" if arm else "b2_velocity"
      raise RuntimeError(
        f"policy not found: {policy_file}\n"
        f"Train one with:  python scripts/train.py {task}  (unitree_rl_mjlab)\n"
        f"then copy its logs/rsl_rl/{experiment}/<run>/policy.onnx to that path."
      )
    self._load_policy(policy_file)

    self.dt = 1.0 / float(self.get_parameter("control_rate_hz").value)
    self.gait_period = float(self.get_parameter("gait_period").value)
    self.cmd_timeout = float(self.get_parameter("cmd_vel_timeout").value)

    self._lock = threading.Lock()
    self._state: LowState | None = None
    self._cmd = np.zeros(3)
    self._cmd_stamp = 0.0
    self._last_action = np.zeros(NUM_MOTORS, dtype=np.float32)
    self._phase_step = 0
    self._mode = "damping"
    self._mode_t = 0.0
    self._stand_start_q: np.ndarray | None = None

    sensor_qos = QoSProfile(
      history=QoSHistoryPolicy.KEEP_LAST,
      depth=1,
      reliability=QoSReliabilityPolicy.BEST_EFFORT,
    )
    self.low_cmd_pub = self.create_publisher(
      LowCmd, self.get_parameter("low_cmd_topic").value, sensor_qos
    )
    self.create_subscription(
      LowState, self.get_parameter("low_state_topic").value, self._on_low_state, sensor_qos
    )
    self.create_subscription(
      Twist, self.get_parameter("cmd_vel_topic").value, self._on_cmd_vel, 10
    )
    self.create_subscription(String, "/b2/mode", self._on_mode, 10)
    # Latched: a late subscriber still sees which state the FSM is in.
    self.mode_pub = self.create_publisher(
      String,
      "/b2/mode_state",
      QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        reliability=QoSReliabilityPolicy.RELIABLE,
      ),
    )
    self.mode_pub.publish(String(data=self._mode))

    self.create_timer(self.dt, self._control_step)

    self.get_logger().info(
      f"B2 policy runner up: {Path(policy_file).name}, "
      f"{self.obs_dim} obs -> {NUM_MOTORS} actions @ "
      f"{1.0 / self.dt:.0f} Hz, gait period {self.gait_period:.2f} s"
    )

  def _load_policy(self, policy_file: str) -> None:
    self.session = ort.InferenceSession(policy_file, providers=["CPUExecutionProvider"])
    self.input_name = self.session.get_inputs()[0].name
    self.obs_dim = int(self.session.get_inputs()[0].shape[-1])

    meta = self.session.get_modelmeta().custom_metadata_map
    missing = {"joint_names", "default_joint_pos", "joint_stiffness",
               "joint_damping", "action_scale"} - meta.keys()
    if missing:
      raise RuntimeError(f"policy ONNX is missing metadata: {sorted(missing)}")

    obs_names = tuple(meta.get("observation_names", "").split(","))
    if obs_names != EXPECTED_OBS:
      raise RuntimeError(
        f"policy observation layout {obs_names} does not match this runner's "
        f"{EXPECTED_OBS}; the runner would feed it garbage."
      )

    policy_joints = meta["joint_names"].split(",")
    if sorted(policy_joints) != sorted(SDK_JOINT_NAMES):
      raise RuntimeError(f"policy joint set {policy_joints} is not B2's 12 joints")

    # The policy uses MuJoCo's joint order (FL, FR, RL, RR); LowState/LowCmd use the
    # SDK's (FR, FL, RR, RL).  These two index maps are the only place that matters.
    self.sdk_to_policy = np.array(
      [SDK_JOINT_NAMES.index(j) for j in policy_joints], dtype=int
    )
    self.policy_to_sdk = np.argsort(self.sdk_to_policy)

    def as_array(key: str) -> np.ndarray:
      return np.array([float(v) for v in meta[key].split(",")])

    self.default_q_policy = as_array("default_joint_pos")
    self.kp_policy = as_array("joint_stiffness")
    self.kd_policy = as_array("joint_damping")

    scale = meta["action_scale"]
    self.action_scale = (
      np.array([float(v) for v in scale.split(",")])
      if "," in scale
      else float(scale)
    )

    # Gains reindexed once into SDK order for command assembly.
    self.kp_sdk = self.kp_policy[self.policy_to_sdk]
    self.kd_sdk = self.kd_policy[self.policy_to_sdk]

  def _on_low_state(self, msg: LowState) -> None:
    with self._lock:
      self._state = msg

  def _on_cmd_vel(self, msg: Twist) -> None:
    with self._lock:
      self._cmd = np.array([
        np.clip(msg.linear.x,
                self.get_parameter("min_lin_vel_x").value,
                self.get_parameter("max_lin_vel_x").value),
        np.clip(msg.linear.y,
                -self.get_parameter("max_lin_vel_y").value,
                self.get_parameter("max_lin_vel_y").value),
        np.clip(msg.angular.z,
                -self.get_parameter("max_ang_vel_z").value,
                self.get_parameter("max_ang_vel_z").value),
      ])
      self._cmd_stamp = self._now()

  def _on_mode(self, msg: String) -> None:
    mode = msg.data.strip().lower()
    if mode not in ("damping", "stand", "policy"):
      self.get_logger().warn(f"ignoring unknown mode '{msg.data}'")
      return
    self._enter_mode(mode)

  def _now(self) -> float:
    return self.get_clock().now().nanoseconds * 1e-9

  def _enter_mode(self, mode: str) -> None:
    self._mode = mode
    self._mode_t = 0.0
    if mode == "stand":
      self._stand_start_q = None
    if mode == "policy":
      self._phase_step = 0
      self._last_action[:] = 0.0
    self.get_logger().info(f"mode -> {mode}")
    self.mode_pub.publish(String(data=mode))

  def _control_step(self) -> None:
    with self._lock:
      state = self._state
      cmd = self._cmd.copy()
      cmd_stamp = self._cmd_stamp
    if state is None:
      return

    self._mode_t += self.dt
    if self._now() - cmd_stamp > self.cmd_timeout:
      cmd = np.zeros(3)

    q_sdk = np.array([state.motor_state[i].q for i in range(NUM_MOTORS)])
    dq_sdk = np.array([state.motor_state[i].dq for i in range(NUM_MOTORS)])

    if self._mode == "damping":
      self._publish_cmd(q_sdk, np.zeros(NUM_MOTORS), np.full(NUM_MOTORS, DAMPING_KD))
      if (
        self.get_parameter("auto_start").value
        and self._mode_t >= float(self.get_parameter("damping_duration").value)
      ):
        self._enter_mode("stand")
      return

    if self._mode == "stand":
      self._stand_step(q_sdk)
      return

    self._policy_step(state, q_sdk, dq_sdk, cmd)

  def _stand_step(self, q_sdk: np.ndarray) -> None:
    """Scripted fold-and-stand, interpolated from the pose we started in."""
    if self._stand_start_q is None:
      self._stand_start_q = q_sdk.copy()

    t = self._mode_t
    prev_t, prev_q = 0.0, self._stand_start_q
    target = np.tile(STAND_KEYFRAMES[-1][1], 4)
    for key_t, leg_q in STAND_KEYFRAMES:
      key_q = np.tile(leg_q, 4)
      if t <= key_t:
        alpha = (t - prev_t) / max(key_t - prev_t, 1e-6)
        target = prev_q + alpha * (key_q - prev_q)
        break
      prev_t, prev_q = key_t, key_q

    self._publish_cmd(
      target, np.full(NUM_MOTORS, STAND_KP), np.full(NUM_MOTORS, STAND_KD)
    )

    if t >= STAND_KEYFRAMES[-1][0] + 0.5 and self.get_parameter("auto_start").value:
      self._enter_mode("policy")

  def _policy_step(
    self, state: LowState, q_sdk: np.ndarray, dq_sdk: np.ndarray, cmd: np.ndarray
  ) -> None:
    gyro = np.asarray(state.imu_state.gyroscope, dtype=np.float64)
    grav = _quat_to_projected_gravity(
      np.asarray(state.imu_state.quaternion, dtype=np.float64)
    )

    # Training zeroes the gait phase whenever the command is below the standing
    # threshold, so the policy learns a distinct stand-still behaviour.
    if np.linalg.norm(cmd) < 0.1:
      phase = np.zeros(2)
    else:
      theta = 2.0 * math.pi * (
        (self._phase_step * self.dt) % self.gait_period / self.gait_period
      )
      phase = np.array([math.sin(theta), math.cos(theta)])
    self._phase_step += 1

    q_policy = q_sdk[self.sdk_to_policy]
    dq_policy = dq_sdk[self.sdk_to_policy]

    obs = np.concatenate((
      gyro,
      grav,
      cmd,
      phase,
      q_policy - self.default_q_policy,
      dq_policy,
      self._last_action,
    )).astype(np.float32)

    if obs.shape[0] != self.obs_dim:
      self.get_logger().error(
        f"built {obs.shape[0]} observations but the policy wants {self.obs_dim}"
      )
      return

    action = self.session.run(None, {self.input_name: obs[None, :]})[0][0]
    self._last_action = action.astype(np.float32)

    q_des_policy = self.default_q_policy + self.action_scale * action
    self._publish_cmd(q_des_policy[self.policy_to_sdk], self.kp_sdk, self.kd_sdk)

  def _publish_cmd(self, q_sdk: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
    msg = LowCmd()
    msg.head = [0xFE, 0xEF]
    msg.level_flag = 0xFF
    for i in range(NUM_MOTORS):
      motor = msg.motor_cmd[i]
      motor.mode = 0x01
      motor.q = float(q_sdk[i])
      motor.dq = 0.0
      motor.kp = float(kp[i])
      motor.kd = float(kd[i])
      motor.tau = 0.0
    # Unused motor slots stay disabled.
    for i in range(NUM_MOTORS, 20):
      msg.motor_cmd[i].mode = 0x00
    self.low_cmd_pub.publish(msg)


def main() -> None:
  rclpy.init()
  node = B2PolicyRunner()
  try:
    rclpy.spin(node)
  except (KeyboardInterrupt, ExternalShutdownException):
    pass
  finally:
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
  main()
