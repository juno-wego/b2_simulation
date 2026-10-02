"""Front-camera geometry and worker tests, with optional real EGL rendering.

Run from the package directory with a Python environment containing MuJoCo:
    python -m pytest test/test_front_camera.py
The worker tests replace only the GL renderer; they still use real MjData.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from b2_mujoco import front_camera
from b2_mujoco.front_camera import (
  CAMERA_NAME,
  FrontCameraConfig,
  LatestFrameRenderer,
  load_model,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]
SCENE_FILE = PACKAGE_DIR / "models" / "b2_nav_scene.xml"


@pytest.fixture(scope="module")
def baseline_model():
  return mujoco.MjModel.from_xml_path(str(SCENE_FILE))


@pytest.fixture(scope="module")
def camera_model():
  return load_model(str(SCENE_FILE), FrontCameraConfig())


@pytest.fixture
def tiny_model():
  return mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><body name="base_link">'
    '<freejoint/><geom type="sphere" size="0.1"/>'
    f'<camera name="{CAMERA_NAME}" pos="0.2 0 0" '
    'xyaxes="0 -1 0 0 0 1"/>'
    '</body><body name="target" mocap="true" pos="1 0 0">'
    '<geom type="sphere" size="0.1"/>'
    '</body></worldbody></mujoco>'
  )


def test_nominal_configuration_and_intrinsics():
  config = FrontCameraConfig()
  assert (config.width, config.height, config.rate_hz, config.fovy_deg) == (
    640, 480, 10.0, 60.0
  )
  assert config.offset == (0.39079, 0.0, -0.013433)
  focal_length = 240.0 / math.tan(math.radians(30.0))
  assert config.focal_length_px == pytest.approx(focal_length)
  assert config.intrinsics == pytest.approx(
    (focal_length, focal_length, 320.0, 240.0)
  )


def test_vertical_fov_gives_square_pixels_at_different_aspect_ratios():
  narrow = FrontCameraConfig(width=320, height=480, fovy_deg=80.0)
  wide = FrontCameraConfig(width=1280, height=480, fovy_deg=80.0)
  expected = 240.0 / math.tan(math.radians(40.0))
  assert narrow.intrinsics == pytest.approx((expected, expected, 160.0, 240.0))
  assert wide.intrinsics == pytest.approx((expected, expected, 640.0, 240.0))


@pytest.mark.parametrize("kwargs", [
  {"width": 0},
  {"width": -1},
  {"width": 640.5},
  {"height": 0},
  {"height": -1},
  {"rate_hz": 0.0},
  {"rate_hz": -1.0},
  {"rate_hz": float("nan")},
  {"rate_hz": float("inf")},
  {"fovy_deg": 0.0},
  {"fovy_deg": 180.0},
  {"fovy_deg": float("nan")},
  {"fovy_deg": float("inf")},
  {"offset": (0.0, 0.0)},
  {"offset": (0.0, 0.0, 0.0, 0.0)},
  {"offset": (0.0, float("nan"), 0.0)},
  {"offset": (0.0, 0.0, float("inf"))},
])
def test_invalid_configuration_is_rejected(kwargs):
  with pytest.raises((TypeError, ValueError)):
    FrontCameraConfig(**kwargs)


def test_disabled_camera_uses_original_model_loader(monkeypatch):
  sentinel = object()
  calls = []

  def from_xml_path(path):
    calls.append(path)
    return sentinel

  monkeypatch.setattr(mujoco.MjModel, "from_xml_path", from_xml_path)
  assert load_model(str(SCENE_FILE), None) is sentinel
  assert calls == [str(SCENE_FILE)]


def test_camera_addition_preserves_robot_model(baseline_model, camera_model):
  assert camera_model.ncam == baseline_model.ncam + 1
  for dimension in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nsite",
                    "nsensor", "nkey", "nmocap"):
    assert getattr(camera_model, dimension) == getattr(baseline_model, dimension)
  for field in (
    "qpos0", "body_mass", "body_inertia", "body_ipos", "body_iquat",
    "dof_damping", "dof_armature", "jnt_type", "jnt_qposadr", "jnt_dofadr",
    "jnt_range", "geom_contype", "geom_conaffinity", "geom_friction",
    "actuator_trnid", "actuator_gear", "actuator_ctrlrange",
    "sensor_type", "sensor_objid", "sensor_adr", "key_qpos", "key_qvel",
    "key_ctrl", "key_time",
  ):
    np.testing.assert_array_equal(
      getattr(camera_model, field), getattr(baseline_model, field), err_msg=field
    )
  assert camera_model.opt.timestep == baseline_model.opt.timestep
  np.testing.assert_array_equal(camera_model.opt.gravity, baseline_model.opt.gravity)
  for obj_type, count in (
    (mujoco.mjtObj.mjOBJ_BODY, baseline_model.nbody),
    (mujoco.mjtObj.mjOBJ_JOINT, baseline_model.njnt),
    (mujoco.mjtObj.mjOBJ_ACTUATOR, baseline_model.nu),
    (mujoco.mjtObj.mjOBJ_KEY, baseline_model.nkey),
  ):
    for index in range(count):
      assert mujoco.mj_id2name(camera_model, obj_type, index) == (
        mujoco.mj_id2name(baseline_model, obj_type, index)
      )


def test_camera_addition_preserves_physics_trajectory(baseline_model, camera_model):
  baseline_data = mujoco.MjData(baseline_model)
  camera_data = mujoco.MjData(camera_model)
  key = mujoco.mj_name2id(baseline_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
  for model, data in ((baseline_model, baseline_data), (camera_model, camera_data)):
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = np.linspace(-2.0, 2.0, model.nu)
    for _ in range(10):
      mujoco.mj_step(model, data)
  np.testing.assert_array_equal(camera_data.qpos, baseline_data.qpos)
  np.testing.assert_array_equal(camera_data.qvel, baseline_data.qvel)
  np.testing.assert_array_equal(camera_data.sensordata, baseline_data.sensordata)
  assert camera_data.time == baseline_data.time


def test_camera_mount_and_view_axes_follow_base(camera_model):
  camera_id = mujoco.mj_name2id(camera_model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
  base_id = mujoco.mj_name2id(camera_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
  assert camera_id >= 0
  assert camera_model.cam_bodyid[camera_id] == base_id
  np.testing.assert_allclose(camera_model.cam_pos[camera_id], FrontCameraConfig().offset)
  assert camera_model.cam_fovy[camera_id] == pytest.approx(60.0)

  data = mujoco.MjData(camera_model)
  data.qpos[:3] = (3.0, -2.0, 1.0)
  yaw = math.radians(35.0)
  data.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
  mujoco.mj_forward(camera_model, data)
  base_rotation = data.xmat[base_id].reshape(3, 3)
  camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
  # MuJoCo cameras look down -Z, with +X right and +Y up.
  np.testing.assert_allclose(-camera_rotation[:, 2], base_rotation[:, 0], atol=1e-12)
  np.testing.assert_allclose(camera_rotation[:, 0], -base_rotation[:, 1], atol=1e-12)
  np.testing.assert_allclose(camera_rotation[:, 1], base_rotation[:, 2], atol=1e-12)
  np.testing.assert_allclose(
    data.cam_xpos[camera_id],
    data.xpos[base_id] + base_rotation @ np.asarray(FrontCameraConfig().offset),
    atol=1e-12,
  )


@pytest.fixture
def fake_renderer(monkeypatch):
  instances = []

  class RecordingRenderer:
    def __init__(self, model, height, width):
      self.model = model
      self.height = height
      self.width = width
      self.owner = threading.get_ident()
      self.snapshots = []
      self.closed = threading.Event()
      instances.append(self)

    def update_scene(self, data, camera=None, **kwargs):
      assert threading.get_ident() == self.owner
      assert camera == CAMERA_NAME or camera == mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME
      )
      self.snapshots.append({
        "data": data,
        "qpos": data.qpos.copy(),
        "time": float(data.time),
        "mocap_pos": data.mocap_pos.copy(),
        "mocap_quat": data.mocap_quat.copy(),
      })

    def render(self):
      assert threading.get_ident() == self.owner
      return np.full((self.height, self.width, 3), 127, dtype=np.uint8)

    def close(self):
      assert threading.get_ident() == self.owner
      self.closed.set()

  monkeypatch.setattr(front_camera.mujoco, "Renderer", RecordingRenderer)
  return instances


def test_submit_owns_snapshot_and_does_not_mutate_physics_data(tiny_model, fake_renderer):
  data = mujoco.MjData(tiny_model)
  data.time = 1.25
  data.qpos[:3] = (1.0, 2.0, 3.0)
  data.mocap_pos[:] = (4.0, 5.0, 6.0)
  data.mocap_quat[:] = (0.5, 0.5, 0.5, 0.5)
  before_qpos = data.qpos.copy()
  before_mocap_pos = data.mocap_pos.copy()
  before_mocap_quat = data.mocap_quat.copy()
  received = []
  callback_done = threading.Event()

  def callback(rgb, sim_time):
    received.append((rgb, sim_time))
    callback_done.set()

  worker = LatestFrameRenderer(tiny_model, FrontCameraConfig(width=32, height=24), callback)
  worker.start()
  try:
    worker.submit(data)
    # A later physics step must not alter the frame already submitted.
    data.qpos[:3] = (-1.0, -2.0, -3.0)
    data.mocap_pos[:] = (-4.0, -5.0, -6.0)
    data.mocap_quat[:] = (1.0, 0.0, 0.0, 0.0)
    data.time = 9.0
    assert callback_done.wait(3.0)
    worker.raise_if_failed()
    snapshot = fake_renderer[0].snapshots[0]
    assert snapshot["data"] is not data
    np.testing.assert_array_equal(snapshot["qpos"], before_qpos)
    np.testing.assert_array_equal(snapshot["mocap_pos"], before_mocap_pos)
    np.testing.assert_array_equal(snapshot["mocap_quat"], before_mocap_quat)
    assert snapshot["time"] == received[0][1] == 1.25
    assert received[0][0].shape == (24, 32, 3)
    assert received[0][0].dtype == np.uint8
    np.testing.assert_array_equal(data.qpos[:3], (-1.0, -2.0, -3.0))
    np.testing.assert_array_equal(data.mocap_pos[0], (-4.0, -5.0, -6.0))
    assert data.time == 9.0
  finally:
    worker.stop()
  assert fake_renderer[0].closed.is_set()
  assert fake_renderer[0].owner != threading.get_ident()


def test_busy_renderer_keeps_only_latest_pending_frame(tiny_model, fake_renderer):
  data = mujoco.MjData(tiny_model)
  first_callback = threading.Event()
  release_callback = threading.Event()
  newest_callback = threading.Event()
  timestamps = []

  def callback(rgb, sim_time):
    timestamps.append(sim_time)
    if len(timestamps) == 1:
      first_callback.set()
      if not release_callback.wait(3.0):
        raise TimeoutError("test did not release first callback")
    else:
      newest_callback.set()

  worker = LatestFrameRenderer(tiny_model, FrontCameraConfig(width=32, height=24), callback)
  worker.start()
  try:
    data.time = 1.0
    worker.submit(data)
    assert first_callback.wait(3.0)
    for sim_time in (2.0, 3.0):
      data.time = sim_time
      data.qpos[0] = sim_time
      worker.submit(data)
    release_callback.set()
    assert newest_callback.wait(3.0)
    worker.raise_if_failed()
  finally:
    release_callback.set()
    worker.stop()
  assert timestamps == [1.0, 3.0]
  assert [frame["time"] for frame in fake_renderer[0].snapshots] == timestamps
  assert fake_renderer[0].snapshots[1]["qpos"][0] == 3.0


def test_renderer_initialization_error_reaches_caller(tiny_model, monkeypatch):
  failure = RuntimeError("test GL initialization failed")

  def failing_renderer(*args, **kwargs):
    raise failure

  monkeypatch.setattr(front_camera.mujoco, "Renderer", failing_renderer)
  worker = LatestFrameRenderer(tiny_model, FrontCameraConfig(width=32, height=24), lambda *_: None)
  with pytest.raises(RuntimeError, match="camera|render|initial"):
    worker.start(timeout=3.0)
  assert worker.error is failure
  worker.stop()


def test_callback_error_is_reported_and_renderer_is_closed(tiny_model, fake_renderer):
  failure = ValueError("test frame callback failed")

  def callback(*args):
    raise failure

  worker = LatestFrameRenderer(tiny_model, FrontCameraConfig(width=32, height=24), callback)
  worker.start()
  try:
    worker.submit(mujoco.MjData(tiny_model))
    assert fake_renderer[0].closed.wait(3.0)
    assert worker.error is failure
    with pytest.raises(RuntimeError):
      worker.raise_if_failed()
  finally:
    worker.stop()


def test_idle_stop_is_idempotent(tiny_model, fake_renderer):
  worker = LatestFrameRenderer(tiny_model, FrontCameraConfig(width=32, height=24), lambda *_: None)
  worker.start()
  worker.stop()
  worker.stop()
  assert fake_renderer[0].closed.is_set()
  assert fake_renderer[0].snapshots == []
  assert worker.error is None


def test_callback_can_stop_its_own_worker(tiny_model, fake_renderer):
  callback_done = threading.Event()

  def callback(*args):
    worker.stop()
    callback_done.set()

  worker = LatestFrameRenderer(tiny_model, FrontCameraConfig(width=32, height=24), callback)
  worker.start()
  try:
    worker.submit(mujoco.MjData(tiny_model))
    assert callback_done.wait(3.0)
  finally:
    worker.stop()
  assert worker.error is None
  assert fake_renderer[0].closed.is_set()


def test_start_timeout_stops_late_context_on_its_owner_thread(
  tiny_model, fake_renderer, monkeypatch,
):
  release_initialization = threading.Event()
  recording_renderer = front_camera.mujoco.Renderer

  def slow_renderer(*args, **kwargs):
    if not release_initialization.wait(3.0):
      raise TimeoutError("test did not release renderer initialization")
    return recording_renderer(*args, **kwargs)

  monkeypatch.setattr(front_camera.mujoco, "Renderer", slow_renderer)
  worker = LatestFrameRenderer(tiny_model, FrontCameraConfig(width=32, height=24), lambda *_: None)
  try:
    with pytest.raises(TimeoutError, match="initialization"):
      worker.start(timeout=0.01)
    with pytest.raises(RuntimeError, match="running"):
      worker.submit(mujoco.MjData(tiny_model))
  finally:
    release_initialization.set()
    worker.stop(timeout=3.0)
  assert len(fake_renderer) == 1
  assert fake_renderer[0].closed.is_set()
  assert fake_renderer[0].snapshots == []
  assert worker.error is None


def test_submission_requires_a_running_worker(tiny_model, fake_renderer):
  worker = LatestFrameRenderer(tiny_model, FrontCameraConfig(width=32, height=24), lambda *_: None)
  data = mujoco.MjData(tiny_model)
  with pytest.raises(RuntimeError, match="running"):
    worker.submit(data)
  worker.start()
  worker.stop()
  with pytest.raises(RuntimeError, match="running"):
    worker.submit(data)


@pytest.fixture
def sim_node_module():
  # Importing messages and the class does not initialize a ROS context or node.
  # Standalone MuJoCo installations can still run all the geometry/worker tests.
  return pytest.importorskip("b2_mujoco.sim_node")


def test_ros_image_and_calibration_share_capture_time(sim_node_module, monkeypatch):
  config = FrontCameraConfig(width=6, height=4, fovy_deg=70.0)
  images = []
  calibration = []
  context = object()
  node = SimpleNamespace(
    _running=True,
    _front_camera_config=config,
    _front_camera_optical_frame="test/camera_optical",
    context=context,
    front_image_pub=SimpleNamespace(publish=images.append),
    front_info_pub=SimpleNamespace(publish=calibration.append),
  )
  monkeypatch.setattr(sim_node_module.rclpy, "ok", lambda **kwargs: kwargs["context"] is context)
  # A non-contiguous RGB view must still produce tightly packed rows.
  rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)[:, ::-1]
  sim_node_module.B2MujocoSim._publish_front_camera(node, rgb, 12.125)
  assert len(images) == len(calibration) == 1
  image, info = images[0], calibration[0]
  assert image.header == info.header
  assert image.header.frame_id == "test/camera_optical"
  assert (image.header.stamp.sec, image.header.stamp.nanosec) == (12, 125000000)
  assert (image.width, image.height, image.encoding, image.step) == (6, 4, "rgb8", 18)
  assert bytes(image.data) == rgb.tobytes()
  assert image.is_bigendian == 0
  assert (info.width, info.height, info.distortion_model) == (6, 4, "plumb_bob")
  np.testing.assert_array_equal(info.d, np.zeros(5))
  fx, fy, cx, cy = config.intrinsics
  np.testing.assert_allclose(info.k, [fx, 0, cx, 0, fy, cy, 0, 0, 1])
  np.testing.assert_array_equal(info.r, np.eye(3).ravel())
  np.testing.assert_allclose(info.p, [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0])


@pytest.mark.parametrize("running, context_ok", [(False, True), (True, False)])
def test_camera_callback_does_not_publish_during_shutdown(
  sim_node_module, monkeypatch, running, context_ok,
):
  messages = []
  node = SimpleNamespace(
    _running=running,
    context=object(),
    front_image_pub=SimpleNamespace(publish=messages.append),
    front_info_pub=SimpleNamespace(publish=messages.append),
  )
  monkeypatch.setattr(sim_node_module.rclpy, "ok", lambda **kwargs: context_ok)
  sim_node_module.B2MujocoSim._publish_front_camera(node, np.zeros((1, 1, 3), np.uint8), 0.0)
  assert messages == []


@pytest.mark.parametrize("enabled", [False, True])
def test_static_camera_tf_chain_and_optical_axes(sim_node_module, enabled):
  transforms_sent = []
  parameters = {
    "base_frame": "test/base",
    "lidar_frame": "test/lidar",
    "lidar_offset": (0.3, 0.0, 0.2),
    "front_camera_link": "test/camera",
    "front_camera_optical_frame": "test/camera_optical",
  }
  config = FrontCameraConfig(offset=(0.4, -0.02, 0.1)) if enabled else None
  node = SimpleNamespace(
    _front_camera_config=config,
    _front_camera_link="test/camera",
    _front_camera_optical_frame="test/camera_optical",
    get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    static_tf_broadcaster=SimpleNamespace(sendTransform=transforms_sent.append),
  )
  sim_node_module.B2MujocoSim._publish_static_tf(node)
  assert len(transforms_sent) == 1
  transforms = transforms_sent[0]
  assert len(transforms) == (3 if enabled else 1)
  assert (transforms[0].header.frame_id, transforms[0].child_frame_id) == (
    "test/base", "test/lidar"
  )
  if not enabled:
    return
  mount, optical = transforms[1:]
  assert (mount.header.frame_id, mount.child_frame_id) == ("test/base", "test/camera")
  assert (optical.header.frame_id, optical.child_frame_id) == ("test/camera", "test/camera_optical")
  translation = mount.transform.translation
  np.testing.assert_array_equal((translation.x, translation.y, translation.z), config.offset)
  assert mount.transform.rotation.w == 1.0
  rotation = optical.transform.rotation
  rotation_matrix = np.empty(9)
  mujoco.mju_quat2Mat(
    rotation_matrix, np.array((rotation.w, rotation.x, rotation.y, rotation.z))
  )
  # ROS optical +X right=-baseY, +Y down=-baseZ, +Z forward=baseX.
  np.testing.assert_array_equal(
    rotation_matrix.reshape(3, 3), [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
  )
  assert optical.transform.translation.x == 0.0
  assert optical.transform.translation.y == 0.0
  assert optical.transform.translation.z == 0.0


@pytest.mark.parametrize("physics_stops", [True, False])
def test_node_teardown_stops_workers_before_destroying_ros_resources(
  sim_node_module, monkeypatch, physics_stops,
):
  calls = []

  class PhysicsThread:
    alive = True

    def is_alive(self):
      return self.alive

    def join(self, timeout):
      calls.append("join physics")
      assert not node._running
      if physics_stops:
        self.alive = False

  node = object.__new__(sim_node_module.B2MujocoSim)
  node._running = True
  node._thread = PhysicsThread()
  node._front_camera_renderer = SimpleNamespace(stop=lambda: calls.append("stop camera"))
  node._viewer = SimpleNamespace(close=lambda: calls.append("close viewer"))

  def destroy_ros_resources(self):
    calls.append("destroy ROS resources")
    return True

  monkeypatch.setattr(sim_node_module.Node, "destroy_node", destroy_ros_resources)
  if physics_stops:
    assert node.destroy_node() is True
    assert calls == ["join physics", "stop camera", "close viewer", "destroy ROS resources"]
  else:
    with pytest.raises(TimeoutError, match="physics"):
      node.destroy_node()
    assert calls == ["join physics", "stop camera"]


def test_main_shuts_down_context_after_constructor_failure(sim_node_module, monkeypatch):
  calls = []
  failure = RuntimeError("test simulator initialization failed")

  def failed_constructor():
    raise failure

  monkeypatch.setattr(sim_node_module, "B2MujocoSim", failed_constructor)
  monkeypatch.setattr(sim_node_module.rclpy, "init", lambda: calls.append("initialize context"))
  monkeypatch.setattr(sim_node_module.rclpy, "try_shutdown", lambda: calls.append("shutdown context"))
  with pytest.raises(RuntimeError, match="initialization"):
    sim_node_module.main()
  assert calls == ["initialize context", "shutdown context"]


def test_real_egl_rgb_frame_in_subprocess():
  # GL backends are selected at import time. A subprocess keeps this optional
  # hardware check isolated from the fake-renderer unit tests and ROS processes.
  script = r'''
import sys
import threading
try:
    import mujoco
    probe_model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
    probe = mujoco.Renderer(probe_model, height=16, width=16)
    probe.close()
except Exception as exc:
    print(f"EGL unavailable: {exc}")
    raise SystemExit(77)

import numpy as np
from b2_mujoco.front_camera import FrontCameraConfig, LatestFrameRenderer, load_model
config = FrontCameraConfig(width=160, height=120)
model = load_model(sys.argv[1], config)
data = mujoco.MjData(model)
key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'stand')
assert key >= 0
mujoco.mj_resetDataKeyframe(model, data, key)
data.time = 0.125
mujoco.mj_forward(model, data)
frames = []
done = threading.Event()
def receive(rgb, sim_time):
    frames.append((rgb.copy(), sim_time))
    done.set()
worker = LatestFrameRenderer(model, config, receive)
worker.start()
try:
    worker.submit(data)
    assert done.wait(10.0), 'no RGB frame received'
    worker.raise_if_failed()
finally:
    worker.stop()
rgb, sim_time = frames[0]
assert rgb.shape == (120, 160, 3), rgb.shape
assert rgb.dtype == np.uint8, rgb.dtype
assert sim_time == 0.125, sim_time
assert np.ptp(rgb) > 20, 'camera frame has no scene contrast'
print('EGL RGB frame and timestamp verified')

# Independently verify image projection and attachment to a moving base with a
# colored target. This uses pixel color only, with no camera/detector dependency.
from pathlib import Path
import tempfile
config = FrontCameraConfig(width=320, height=240)
mount_x, mount_y, mount_z = config.offset
xml = f"""<mujoco><worldbody>
<body name="base_link"><freejoint/><geom type="sphere" size="0.02" mass="1"/></body>
<geom name="target" type="box" size="0.001 0.1 0.1"
      pos="{mount_x + 2.0} 0.4 {mount_z + 0.2}" rgba="1 0 0 1"/>
</worldbody></mujoco>"""
with tempfile.TemporaryDirectory() as directory:
    scene = Path(directory) / 'projection.xml'
    scene.write_text(xml)
    model = load_model(scene, config)
data = mujoco.MjData(model)
frames = []
done.clear()
worker = LatestFrameRenderer(model, config, receive)
worker.start()
try:
    for base_y, sim_time in ((0.0, 1.0), (0.2, 2.0)):
        done.clear()
        data.qpos[1] = base_y
        data.time = sim_time
        worker.submit(data)
        assert done.wait(10.0), 'no projection frame received'
        worker.raise_if_failed()
finally:
    worker.stop()
centers = []
for rgb, sim_time in frames:
    red, green, blue = rgb.astype(float).transpose(2, 0, 1)
    rows, columns = np.nonzero((red > 80) & (red > green * 1.5) & (red > blue * 1.5))
    assert len(rows) > 20, 'red target not visible'
    centers.append(np.array((columns.mean(), rows.mean())))
focal, _, cx, cy = config.intrinsics
expected = np.array((cx - focal * 0.4 / 2.0, cy - focal * 0.2 / 2.0))
np.testing.assert_allclose(centers[0], expected, atol=2.0)
np.testing.assert_allclose(centers[1] - centers[0], (focal * 0.2 / 2.0, 0.0), atol=2.0)
assert [stamp for _, stamp in frames] == [1.0, 2.0]
print('EGL projection and base-motion verified')
'''
  env = os.environ.copy()
  env["MUJOCO_GL"] = "egl"
  env["PYTHONPATH"] = str(PACKAGE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
  result = subprocess.run(
    [sys.executable, "-c", script, str(SCENE_FILE)],
    capture_output=True, text=True, timeout=30.0, env=env,
  )
  if result.returncode == 77:
    pytest.skip(result.stdout.strip() or result.stderr.strip())
  assert result.returncode == 0, result.stdout + result.stderr
