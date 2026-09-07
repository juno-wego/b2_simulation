#!/usr/bin/env python3
"""MuJoCo simulator for the Unitree B2, speaking the robot's own low-level interface.

The node loads Unitree's official B2 model and exchanges exactly the messages the
real B2 does -- `unitree_go/LowState` out, `unitree_go/LowCmd` in -- so the policy
runner driving this simulator is the same code that would drive the hardware.
On top of that it publishes what the physical robot's sensor head provides and
`unitree_mujoco` does not: a raycast 3D LiDAR cloud, joint states and `/clock`.

Motor indices follow the Unitree SDK order (FR, FL, RR, RL), which is also the
actuator order of Unitree's B2 MJCF, so no remapping happens here.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

# Must precede the imports below: it puts the b2sim virtualenv on sys.path
# when the node was launched without sourcing b2_env.sh.
import b2_mujoco._venv  # noqa: F401
import mujoco
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
  QoSDurabilityPolicy,
  QoSHistoryPolicy,
  QoSProfile,
  QoSReliabilityPolicy,
)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import BatteryState, Imu, JointState, PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from unitree_go.msg import LowCmd, LowState

from b2_mujoco.lidar import RaycastLidar

NUM_MOTORS = 12

# Unitree SDK motor order.  Identical to the actuator order in Unitree's B2 MJCF.
JOINT_NAMES = [
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
FOOT_GEOM_ORDER = ("FR", "FL", "RR", "RL")


def _quat_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
  """MuJoCo (w, x, y, z) quaternion to roll/pitch/yaw."""
  w, x, y, z = q
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return float(roll), float(pitch), float(yaw)


class B2MujocoSim(Node):
  def __init__(self) -> None:
    super().__init__("b2_mujoco_sim")

    default_scene = str(
      Path(__file__).resolve().parents[1] / "models" / "b2_nav_scene.xml"
    )
    self.declare_parameter("scene_file", default_scene)
    self.declare_parameter("low_state_topic", "/lowstate")
    self.declare_parameter("low_cmd_topic", "/lowcmd")
    self.declare_parameter("points_topic", "/b2/points")
    self.declare_parameter("odom_topic", "/b2/odom_gt")
    self.declare_parameter("imu_topic", "/b2/imu")
    self.declare_parameter("joint_state_topic", "/b2/joint_states")
    self.declare_parameter("base_frame", "base_link")
    self.declare_parameter("lidar_frame", "b2/lidar_link")
    self.declare_parameter("odom_frame", "odom")
    self.declare_parameter("publish_clock", True)
    # Ground-truth odom->base_link TF.  Off by default: KISS-ICP owns that edge in
    # the SLAM stack, and two publishers on one edge corrupt the transform tree.
    self.declare_parameter("publish_ground_truth_tf", False)
    self.declare_parameter("lidar_offset", [0.34218, 0.0, 0.20])
    self.declare_parameter("lidar_rate_hz", 10.0)
    # ground_segmentation subscribes RELIABLE, and a best-effort publisher cannot
    # feed a reliable subscriber -- the cloud would silently never arrive.  Gazebo's
    # ros_gz_bridge publishes reliably for the same reason.
    self.declare_parameter("points_reliable", True)
    self.declare_parameter("lidar_rings", 32)
    self.declare_parameter("lidar_azimuth", 720)
    self.declare_parameter("lidar_fov_lower_deg", -30.0)
    self.declare_parameter("lidar_fov_upper_deg", 12.0)
    self.declare_parameter("lidar_range_max", 30.0)
    self.declare_parameter("state_rate_hz", 200.0)
    self.declare_parameter("battery_topic", "/b2/battery_state")
    self.declare_parameter("arm_state_topic", "/fr3/joint_states")
    self.declare_parameter("arm_command_topic", "/fr3/joint_command")
    # B2 ships a 58 V 45 Ah pack. Capacity and idle draw are the only figures
    # here that are not measured from the simulation itself; everything else
    # comes out of MuJoCo. See _publish_battery.
    self.declare_parameter("battery_wh", 2610.0)
    self.declare_parameter("battery_nominal_v", 58.0)
    self.declare_parameter("battery_idle_w", 120.0)
    self.declare_parameter("battery_efficiency", 0.75)
    self.declare_parameter("battery_start_soc", 0.92)
    # Until a controller connects there is no /lowcmd, and an unpowered 83 kg B2
    # would just collapse into a heap the stand-up script cannot recover from.
    # Hold the spawn pose instead: it stands in for the physical robot already
    # resting on its feet when you plug the controller in.
    self.declare_parameter("startup_hold_kp", 400.0)
    self.declare_parameter("startup_hold_kd", 10.0)
    self.declare_parameter("realtime_factor", 1.0)
    self.declare_parameter("viewer", True)

    scene_file = self.get_parameter("scene_file").value
    self.model = mujoco.MjModel.from_xml_path(scene_file)
    self.data = mujoco.MjData(self.model)
    self.dt = float(self.model.opt.timestep)

    key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key < 0:
      raise RuntimeError(f"{scene_file} has no 'stand' keyframe")
    mujoco.mj_resetDataKeyframe(self.model, self.data, key)
    mujoco.mj_forward(self.model, self.data)

    self._resolve_model_indices()

    self.lidar = RaycastLidar(
      num_rings=int(self.get_parameter("lidar_rings").value),
      num_azimuth=int(self.get_parameter("lidar_azimuth").value),
      fov_lower_deg=float(self.get_parameter("lidar_fov_lower_deg").value),
      fov_upper_deg=float(self.get_parameter("lidar_fov_upper_deg").value),
      range_max=float(self.get_parameter("lidar_range_max").value),
    )
    self.lidar_offset = np.asarray(
      self.get_parameter("lidar_offset").value, dtype=np.float64
    )

    # Command state, written by the /lowcmd callback and read by the physics thread.
    self._cmd_lock = threading.Lock()
    self._spawn_q = self.data.qpos[self._qpos_adr].copy()
    self._hold_kp = np.full(
      NUM_MOTORS, float(self.get_parameter("startup_hold_kp").value)
    )
    self._hold_kd = np.full(
      NUM_MOTORS, float(self.get_parameter("startup_hold_kd").value)
    )
    self._q_des = self._spawn_q.copy()
    self._dq_des = np.zeros(NUM_MOTORS)
    self._kp = np.zeros(NUM_MOTORS)
    self._kd = np.zeros(NUM_MOTORS)
    self._tau_ff = np.zeros(NUM_MOTORS)
    self._got_cmd = False

    sensor_qos = QoSProfile(
      history=QoSHistoryPolicy.KEEP_LAST,
      depth=1,
      reliability=QoSReliabilityPolicy.BEST_EFFORT,
    )

    self.low_state_pub = self.create_publisher(
      LowState, self.get_parameter("low_state_topic").value, sensor_qos
    )
    points_qos = QoSProfile(
      history=QoSHistoryPolicy.KEEP_LAST,
      depth=5,
      reliability=(
        QoSReliabilityPolicy.RELIABLE
        if self.get_parameter("points_reliable").value
        else QoSReliabilityPolicy.BEST_EFFORT
      ),
    )
    self.points_pub = self.create_publisher(
      PointCloud2, self.get_parameter("points_topic").value, points_qos
    )
    self.odom_pub = self.create_publisher(
      Odometry, self.get_parameter("odom_topic").value, sensor_qos
    )
    self.imu_pub = self.create_publisher(
      Imu, self.get_parameter("imu_topic").value, sensor_qos
    )
    self.joint_pub = self.create_publisher(
      JointState, self.get_parameter("joint_state_topic").value, sensor_qos
    )
    # Latched: a control station that connects late should not sit with an
    # empty gauge until the next second ticks.
    self.battery_pub = self.create_publisher(
      BatteryState,
      self.get_parameter("battery_topic").value,
      QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
      ),
    )
    # The arm is only present in the scene that carries it. Everything below
    # checks `self._arm_joints` and does nothing when the robot is bare, so one
    # node serves both scenes - the control station cannot tell which is
    # running except by the arm channel falling silent, which state/health
    # already reports.
    self._arm_joints = [
      j for j in (
        mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(self.model.njnt)
      ) if j and j.startswith("fr3_")
    ]
    self._arm_qadr = [
      self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
      for j in self._arm_joints
    ]
    self._arm_vadr = [
      self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
      for j in self._arm_joints
    ]
    self._arm_actadr = [
      mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
      for j in self._arm_joints
    ]
    self._arm_target = [float(self.data.qpos[a]) for a in self._arm_qadr]

    self.arm_state_pub = None
    if self._arm_joints:
      # RELIABLE, not the sensor profile the rest of this node uses. The
      # bridge and MoveIt2 both subscribe reliably, and a BEST_EFFORT publisher
      # is simply invisible to them — the same QoS mismatch that once left the
      # ground segmenter with no point cloud and no error.
      self.arm_state_pub = self.create_publisher(
        JointState,
        self.get_parameter("arm_state_topic").value,
        QoSProfile(
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=10,
          reliability=QoSReliabilityPolicy.RELIABLE,
        ),
      )
      self.create_subscription(
        JointState,
        self.get_parameter("arm_command_topic").value,
        self._on_arm_command,
        10,
      )
      self.get_logger().info(
        f"FR3 팔 {len(self._arm_joints)}축: "
        f"{self.get_parameter('arm_command_topic').value} -> "
        f"{self.get_parameter('arm_state_topic').value}"
      )

    self.clock_pub = (
      self.create_publisher(Clock, "/clock", 10)
      if self.get_parameter("publish_clock").value
      else None
    )

    self.create_subscription(
      LowCmd, self.get_parameter("low_cmd_topic").value, self._on_low_cmd, sensor_qos
    )

    self.tf_broadcaster = TransformBroadcaster(self)
    self.static_tf_broadcaster = StaticTransformBroadcaster(self)
    self._publish_static_tf()

    self.base_frame = self.get_parameter("base_frame").value
    self.lidar_frame = self.get_parameter("lidar_frame").value
    self.odom_frame = self.get_parameter("odom_frame").value
    self.publish_gt_tf = bool(self.get_parameter("publish_ground_truth_tf").value)

    self._state_decim = max(
      1, round(1.0 / (self.dt * float(self.get_parameter("state_rate_hz").value)))
    )
    self._lidar_decim = max(
      1, round(1.0 / (self.dt * float(self.get_parameter("lidar_rate_hz").value)))
    )
    # The pack moves slowly; 1 Hz is what the protocol asks for.
    self._battery_decim = max(1, round(1.0 / self.dt))

    self._battery_wh = float(self.get_parameter("battery_wh").value)
    self._battery_nominal_v = float(self.get_parameter("battery_nominal_v").value)
    self._battery_idle_w = float(self.get_parameter("battery_idle_w").value)
    self._battery_eff = max(0.05, float(self.get_parameter("battery_efficiency").value))
    self._battery_wh_left = self._battery_wh * float(
      self.get_parameter("battery_start_soc").value
    )
    self._battery_draw_w = 0.0

    self._viewer = None
    if self.get_parameter("viewer").value:
      # Imported lazily so a headless run needs no windowing system.  Bound under
      # its own name: `import mujoco.viewer` here would make `mujoco` a local.
      from mujoco import viewer as mj_viewer

      self._viewer = mj_viewer.launch_passive(
        self.model, self.data, show_left_ui=False, show_right_ui=False
      )

    self._running = True
    self._thread = threading.Thread(target=self._physics_loop, daemon=True)
    self._thread.start()

    self.get_logger().info(
      f"B2 MuJoCo sim up: scene={Path(scene_file).name} dt={self.dt * 1000:.1f} ms, "
      f"LiDAR {self.lidar.num_rays} rays @ "
      f"{self.get_parameter('lidar_rate_hz').value:.0f} Hz"
    )

  def _resolve_model_indices(self) -> None:
    """Map SDK motor order onto MuJoCo joint/actuator/sensor addresses."""
    m = self.model

    self._qpos_adr = np.empty(NUM_MOTORS, dtype=int)
    self._qvel_adr = np.empty(NUM_MOTORS, dtype=int)
    self._ctrl_adr = np.empty(NUM_MOTORS, dtype=int)
    for i, joint in enumerate(JOINT_NAMES):
      jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
      if jid < 0:
        raise RuntimeError(f"joint {joint} missing from model")
      self._qpos_adr[i] = m.jnt_qposadr[jid]
      self._qvel_adr[i] = m.jnt_dofadr[jid]
      # Unitree names the actuator after the joint minus the "_joint" suffix.
      aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, joint[: -len("_joint")])
      if aid < 0:
        raise RuntimeError(f"actuator for {joint} missing from model")
      self._ctrl_adr[i] = aid

    # Torque ceilings come from the model's own ctrlrange rather than a constant,
    # so this can never disagree with the MJCF MuJoCo is actually enforcing.
    # (Unitree's own files differ slightly here: the B2 MJCF caps the calf motor
    # at 300 Nm while their IsaacLab B2 config says 320.)
    self._torque_limit = np.abs(m.actuator_ctrlrange[self._ctrl_adr, 1])

    def sensor_slice(name: str) -> slice:
      sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
      if sid < 0:
        raise RuntimeError(f"sensor {name} missing from model")
      adr = m.sensor_adr[sid]
      return slice(adr, adr + m.sensor_dim[sid])

    self._imu_quat = sensor_slice("imu_quat")
    self._imu_gyro = sensor_slice("imu_gyro")
    self._imu_acc = sensor_slice("imu_acc")

    self._base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    self._twist = np.zeros(6)
    # Unitree's MJCF leaves the foot colliders unnamed, so foot contact is
    # attributed at calf-body level.  Only used for the diagnostic foot_force field.
    self._foot_bids = {
      leg: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
      for leg in FOOT_GEOM_ORDER
    }

  def _publish_static_tf(self) -> None:
    tf = TransformStamped()
    tf.header.frame_id = self.get_parameter("base_frame").value
    tf.child_frame_id = self.get_parameter("lidar_frame").value
    offset = self.get_parameter("lidar_offset").value
    tf.transform.translation.x = float(offset[0])
    tf.transform.translation.y = float(offset[1])
    tf.transform.translation.z = float(offset[2])
    tf.transform.rotation.w = 1.0
    self.static_tf_broadcaster.sendTransform(tf)

  def _on_low_cmd(self, msg: LowCmd) -> None:
    q = np.empty(NUM_MOTORS)
    dq = np.empty(NUM_MOTORS)
    kp = np.empty(NUM_MOTORS)
    kd = np.empty(NUM_MOTORS)
    tau = np.empty(NUM_MOTORS)
    for i in range(NUM_MOTORS):
      cmd = msg.motor_cmd[i]
      q[i], dq[i], kp[i], kd[i], tau[i] = cmd.q, cmd.dq, cmd.kp, cmd.kd, cmd.tau
    with self._cmd_lock:
      self._q_des, self._dq_des = q, dq
      self._kp, self._kd, self._tau_ff = kp, kd, tau
      self._got_cmd = True

  def _sim_time(self) -> TimeMsg:
    t = self.data.time
    msg = TimeMsg()
    msg.sec = int(t)
    msg.nanosec = int((t - msg.sec) * 1e9)
    return msg

  def _foot_forces(self) -> list[int]:
    """Normal force magnitude under each foot, in the LowState int16 convention."""
    forces = {leg: 0.0 for leg in FOOT_GEOM_ORDER}
    buf = np.zeros(6)
    for i in range(self.data.ncon):
      con = self.data.contact[i]
      b1 = self.model.geom_bodyid[con.geom1]
      b2 = self.model.geom_bodyid[con.geom2]
      for leg, bid in self._foot_bids.items():
        if bid in (b1, b2):
          mujoco.mj_contactForce(self.model, self.data, i, buf)
          forces[leg] += abs(float(buf[0]))
    return [int(np.clip(forces[leg], 0, 32767)) for leg in FOOT_GEOM_ORDER]

  def _publish_low_state(self) -> None:
    d = self.data
    msg = LowState()
    stamp = self._sim_time()

    quat = d.sensordata[self._imu_quat].copy()
    msg.imu_state.quaternion = [float(v) for v in quat]
    msg.imu_state.gyroscope = [float(v) for v in d.sensordata[self._imu_gyro]]
    msg.imu_state.accelerometer = [float(v) for v in d.sensordata[self._imu_acc]]
    msg.imu_state.rpy = [float(v) for v in _quat_to_rpy(quat)]

    q = d.qpos[self._qpos_adr]
    dq = d.qvel[self._qvel_adr]
    tau = d.actuator_force[self._ctrl_adr]
    for i in range(NUM_MOTORS):
      motor = msg.motor_state[i]
      motor.mode = 1
      motor.q = float(q[i])
      motor.dq = float(dq[i])
      motor.tau_est = float(tau[i])

    msg.foot_force = self._foot_forces()
    msg.tick = int(d.time * 1e3) & 0xFFFFFFFF
    self.low_state_pub.publish(msg)

    imu = Imu()
    imu.header.stamp = stamp
    imu.header.frame_id = self.base_frame
    imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z = (
      float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    )
    gyro = d.sensordata[self._imu_gyro]
    acc = d.sensordata[self._imu_acc]
    imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = map(float, gyro)
    imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = map(float, acc)
    self.imu_pub.publish(imu)

    js = JointState()
    js.header.stamp = stamp
    js.name = JOINT_NAMES
    js.position = [float(v) for v in q]
    js.velocity = [float(v) for v in dq]
    js.effort = [float(v) for v in tau]
    self.joint_pub.publish(js)

    self._publish_odom(stamp)

  def _publish_odom(self, stamp: TimeMsg) -> None:
    d = self.data
    pos = d.xpos[self._base_bid]
    quat = d.xquat[self._base_bid]
    # Body-frame twist at the base frame's own origin.  d.cvel would give the
    # subtree-CoM velocity in world axes instead, which differs by omega x r.
    mujoco.mj_objectVelocity(
      self.model, d, mujoco.mjtObj.mjOBJ_BODY, self._base_bid, self._twist, 1
    )
    ang, lin = self._twist[0:3], self._twist[3:6]

    odom = Odometry()
    odom.header.stamp = stamp
    odom.header.frame_id = self.odom_frame
    odom.child_frame_id = self.base_frame
    odom.pose.pose.position.x = float(pos[0])
    odom.pose.pose.position.y = float(pos[1])
    odom.pose.pose.position.z = float(pos[2])
    odom.pose.pose.orientation.w = float(quat[0])
    odom.pose.pose.orientation.x = float(quat[1])
    odom.pose.pose.orientation.y = float(quat[2])
    odom.pose.pose.orientation.z = float(quat[3])
    odom.twist.twist.linear.x = float(lin[0])
    odom.twist.twist.linear.y = float(lin[1])
    odom.twist.twist.linear.z = float(lin[2])
    odom.twist.twist.angular.x = float(ang[0])
    odom.twist.twist.angular.y = float(ang[1])
    odom.twist.twist.angular.z = float(ang[2])
    self.odom_pub.publish(odom)

    if self.publish_gt_tf:
      tf = TransformStamped()
      tf.header.stamp = stamp
      tf.header.frame_id = self.odom_frame
      tf.child_frame_id = self.base_frame
      tf.transform.translation.x = float(pos[0])
      tf.transform.translation.y = float(pos[1])
      tf.transform.translation.z = float(pos[2])
      tf.transform.rotation.w = float(quat[0])
      tf.transform.rotation.x = float(quat[1])
      tf.transform.rotation.y = float(quat[2])
      tf.transform.rotation.z = float(quat[3])
      self.tf_broadcaster.sendTransform(tf)

  def _publish_points(self) -> None:
    d = self.data
    rot = d.xmat[self._base_bid].reshape(3, 3)
    origin = d.xpos[self._base_bid] + rot @ self.lidar_offset
    pts = self.lidar.scan(self.model, d, origin, rot)

    msg = PointCloud2()
    msg.header.stamp = self._sim_time()
    msg.header.frame_id = self.lidar_frame
    msg.height = 1
    msg.width = pts.shape[0]
    msg.fields = [
      PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
      PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
      PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * pts.shape[0]
    msg.is_dense = True
    msg.data = np.ascontiguousarray(pts).tobytes()
    self.points_pub.publish(msg)

  def _apply_control(self) -> None:
    d = self.data
    with self._cmd_lock:
      if self._got_cmd:
        q_des, dq_des = self._q_des, self._dq_des
        kp, kd, tau_ff = self._kp, self._kd, self._tau_ff
      else:
        q_des, dq_des = self._spawn_q, np.zeros(NUM_MOTORS)
        kp, kd, tau_ff = self._hold_kp, self._hold_kd, np.zeros(NUM_MOTORS)

    q = d.qpos[self._qpos_adr]
    dq = d.qvel[self._qvel_adr]
    # The same law B2's motor controllers run on-board.
    tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
    d.ctrl[self._ctrl_adr] = np.clip(tau, -self._torque_limit, self._torque_limit)

  def _on_arm_command(self, msg: JointState) -> None:
    """Take a joint target for the arm, by name.

    Matched by name rather than by index because the sender is MoveIt2 or a
    teach panel, and neither guarantees the order this model happens to use.
    Positions outside the joint's range are clamped: the real controller
    refuses them, and letting the simulator reach somewhere the robot cannot
    would make the station's preview a lie.
    """
    for name, q in zip(msg.name, msg.position):
        if name not in self._arm_joints:
            continue
        k = self._arm_joints.index(name)
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = self.model.jnt_range[jid]
        self._arm_target[k] = float(min(max(q, lo), hi))

  def _publish_arm_state(self) -> None:
    msg = JointState()
    msg.header.stamp = self._sim_time()
    msg.name = list(self._arm_joints)
    msg.position = [float(self.data.qpos[a]) for a in self._arm_qadr]
    msg.velocity = [float(self.data.qvel[a]) for a in self._arm_vadr]
    msg.effort = [float(self.data.qfrc_actuator[a]) for a in self._arm_vadr]
    self.arm_state_pub.publish(msg)

  def _drain_battery(self) -> None:
    """Integrate one step of energy out of the pack.

    The mechanical power is real: MuJoCo knows the torque it applied and the
    speed each joint turned at, so `|tau . dq|` is what the legs actually cost
    this step. Only three numbers are modelled rather than measured - the pack's
    capacity, the drivetrain efficiency that turns electrical watts into
    mechanical ones, and the idle draw of the computers and sensors. They are
    parameters, and they are the whole model.

    Absolute value on the joint power: regenerative braking exists in these
    motors but a quadruped does not put meaningful energy back, and pretending
    it does would make the pack last longer in simulation than on the robot.
    """
    joint_w = float(np.abs(self.data.qfrc_actuator[6:] * self.data.qvel[6:]).sum())
    draw_w = self._battery_idle_w + joint_w / self._battery_eff
    self._battery_wh_left -= draw_w * self.dt / 3600.0
    self._battery_wh_left = max(0.0, self._battery_wh_left)
    self._battery_draw_w = draw_w

  def _publish_battery(self) -> None:
    soc = self._battery_wh_left / self._battery_wh if self._battery_wh > 0 else 0.0
    msg = BatteryState()
    msg.header.stamp = self._sim_time()
    msg.percentage = float(soc)
    # A 15S lithium pack runs about 3.3 V a cell empty to 4.2 V full. Linear in
    # SOC is not the real curve, but it puts the voltage in the range an
    # operator recognises instead of leaving it at zero.
    msg.voltage = float(15.0 * (3.3 + 0.9 * soc))
    msg.current = float(-self._battery_draw_w / max(msg.voltage, 1.0))
    msg.charge = float(self._battery_wh_left / max(msg.voltage, 1.0))
    msg.capacity = float(self._battery_wh / self._battery_nominal_v)
    msg.design_capacity = msg.capacity
    msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
    msg.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_GOOD
    msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LION
    msg.present = True
    self.battery_pub.publish(msg)

  def _physics_loop(self) -> None:
    rtf = max(1e-3, float(self.get_parameter("realtime_factor").value))
    step = 0
    next_wall = time.perf_counter()
    while self._running:
      self._apply_control()
      for k, a in enumerate(self._arm_actadr):
        self.data.ctrl[a] = self._arm_target[k]
      mujoco.mj_step(self.model, self.data)
      step += 1

      if self.clock_pub is not None:
        clock = Clock()
        clock.clock = self._sim_time()
        self.clock_pub.publish(clock)
      self._drain_battery()
      if step % self._state_decim == 0:
        self._publish_low_state()
      if step % self._battery_decim == 0:
        self._publish_battery()
      if self.arm_state_pub is not None and step % self._state_decim == 0:
        self._publish_arm_state()
      if step % self._lidar_decim == 0:
        self._publish_points()
      if self._viewer is not None and step % 4 == 0:
        if not self._viewer.is_running():
          self._running = False
          rclpy.try_shutdown()
          break
        self._viewer.sync()

      next_wall += self.dt / rtf
      lag = next_wall - time.perf_counter()
      if lag > 0:
        time.sleep(lag)
      else:
        # Fell behind: give up on catching up rather than spiralling.
        next_wall = time.perf_counter()

  def destroy_node(self) -> bool:
    self._running = False
    if self._thread.is_alive():
      self._thread.join(timeout=1.0)
    if self._viewer is not None:
      self._viewer.close()
    return super().destroy_node()


def main() -> None:
  rclpy.init()
  node = B2MujocoSim()
  try:
    rclpy.spin(node)
  except (KeyboardInterrupt, ExternalShutdownException):
    pass
  finally:
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
  main()
