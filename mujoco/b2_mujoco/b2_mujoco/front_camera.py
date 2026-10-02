"""Optional ideal RGB camera and its latest-frame rendering worker.

The camera is added to an in-memory model specification.  Its nominal mount is
the B2 URDF's front camera position; its view faces base +X with base +Z up.
Rendering owns a private MjData and an OpenGL context on one worker thread.
Only ``submit`` touches the physics data, synchronously on the physics thread.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Callable

# This must precede third-party imports when ROS launches outside b2_env.sh.
import b2_mujoco._venv  # noqa: F401
import mujoco
import numpy as np

CAMERA_NAME = "b2_front_camera"
NOMINAL_CAMERA_OFFSET = (0.39079, 0.0, -0.013433)

# Optical x is base -Y (right), optical y is base -Z (down), optical z is
# base +X (forward).  ROS quaternions are ordered (x, y, z, w).
OPTICAL_QUATERNION_XYZW = (-0.5, 0.5, -0.5, 0.5)

# MuJoCo cameras look along their local -Z with local +Y up.  MuJoCo orders
# quaternions (w, x, y, z), unlike ROS optical-frame transforms above.
_CAMERA_QUATERNION_WXYZ = (0.5, 0.5, -0.5, -0.5)


@dataclass(frozen=True)
class FrontCameraConfig:
  """Ideal pinhole camera settings, with vertical field of view in degrees."""

  width: int = 640
  height: int = 480
  rate_hz: float = 10.0
  fovy_deg: float = 60.0
  offset: tuple[float, float, float] = NOMINAL_CAMERA_OFFSET

  def __post_init__(self) -> None:
    for name in ("width", "height"):
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"front camera {name} must be an integer")
      if not 1 <= value <= 4096:
        raise ValueError(f"front camera {name} must be between 1 and 4096")
      object.__setattr__(self, name, int(value))

    for name in ("rate_hz", "fovy_deg"):
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"front camera {name} must be a finite number")
      if not math.isfinite(value):
        raise ValueError(f"front camera {name} must be a finite number")
      object.__setattr__(self, name, float(value))
    if not 0.0 < self.rate_hz <= 200.0:
      raise ValueError("front camera rate_hz must be greater than 0 and at most 200")
    if not 0.0 < self.fovy_deg < 180.0:
      raise ValueError("front camera fovy_deg must be between 0 and 180 degrees")

    try:
      offset = tuple(self.offset)
    except TypeError as exc:
      raise ValueError("front camera offset must contain three finite numbers") from exc
    if len(offset) != 3 or any(
      isinstance(value, bool) or not isinstance(value, Real)
      or not math.isfinite(value) for value in offset
    ):
      raise ValueError("front camera offset must contain three finite numbers")
    object.__setattr__(self, "offset", tuple(float(value) for value in offset))

  @property
  def focal_length_px(self) -> float:
    """Square-pixel focal length derived from image height and vertical FOV."""
    return self.height / (2.0 * math.tan(math.radians(self.fovy_deg) / 2.0))

  @property
  def intrinsics(self) -> tuple[float, float, float, float]:
    """Return ideal ``(fx, fy, cx, cy)``; these are simulation calibration."""
    focal = self.focal_length_px
    return focal, focal, self.width / 2.0, self.height / 2.0


def load_model(
  scene_file: str | Path, config: FrontCameraConfig | None = None,
) -> mujoco.MjModel:
  """Load the original scene, optionally adding a front camera in memory."""
  if config is None:
    return mujoco.MjModel.from_xml_path(str(scene_file))

  spec = mujoco.MjSpec.from_file(str(scene_file))
  base = spec.body("base_link")
  if base is None:
    raise ValueError(f"{scene_file} has no 'base_link' body for the front camera")
  if spec.camera(CAMERA_NAME) is not None:
    raise ValueError(f"{scene_file} already contains camera '{CAMERA_NAME}'")
  base.add_camera(
    name=CAMERA_NAME,
    pos=config.offset,
    quat=_CAMERA_QUATERNION_WXYZ,
    fovy=config.fovy_deg,
  )
  # The scene's default buffer is 640x480.  Expand it only when needed; larger
  # requested images otherwise fail before the first frame is rendered.
  spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, config.width)
  spec.visual.global_.offheight = max(spec.visual.global_.offheight, config.height)
  return spec.compile()


@dataclass(frozen=True)
class _FrameSnapshot:
  qpos: np.ndarray
  sim_time: float
  mocap_pos: np.ndarray
  mocap_quat: np.ndarray


class LatestFrameRenderer:
  """Render the newest submitted pose without queuing older camera frames.

  ``callback(rgb, sim_time)`` runs on the worker.  Call ``submit(data)`` only
  from the physics thread, at the configured camera rate, and poll
  ``raise_if_failed`` from the simulation loop.  ``stop`` joins the worker
  before node/publisher resources are destroyed.  The renderer is single-use.
  """

  def __init__(
    self, model: mujoco.MjModel, config: FrontCameraConfig,
    callback: Callable[[np.ndarray, float], None],
  ) -> None:
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME) < 0:
      raise ValueError(f"model has no '{CAMERA_NAME}' camera")
    self._model = model
    self._config = config
    self._callback = callback
    self._condition = threading.Condition()
    self._ready = threading.Event()
    self._thread: threading.Thread | None = None
    self._latest: _FrameSnapshot | None = None
    self._stopping = False
    self._error: BaseException | None = None

  @property
  def error(self) -> BaseException | None:
    """First rendering, initialization, callback, or context-close failure."""
    with self._condition:
      return self._error

  def raise_if_failed(self) -> None:
    error = self.error
    if error is not None:
      raise RuntimeError(f"front camera rendering failed: {error}") from error

  def start(self, timeout: float = 10.0) -> None:
    """Wait at most ten seconds for the worker to create its OpenGL context."""
    if not math.isfinite(timeout) or timeout <= 0.0:
      raise ValueError("front camera startup timeout must be finite and positive")
    with self._condition:
      if self._stopping:
        raise RuntimeError("front camera renderer has been stopped")
      if self._thread is None:
        self._thread = threading.Thread(
          target=self._run, name="b2-front-camera", daemon=True,
        )
        self._thread.start()
    if not self._ready.wait(min(timeout, 10.0)):
      # A driver can hang while creating a GL context.  Request shutdown now;
      # only that same worker may release the context if creation later ends.
      with self._condition:
        self._stopping = True
        self._condition.notify_all()
      raise TimeoutError("front camera OpenGL initialization timed out")
    self.raise_if_failed()

  def submit(self, data: mujoco.MjData) -> None:
    """Copy a pose now, replacing any frame that has not begun rendering."""
    with self._condition:
      if self._error is not None:
        raise RuntimeError(f"front camera rendering failed: {self._error}") from self._error
      if self._stopping or self._thread is None or not self._ready.is_set():
        raise RuntimeError("front camera renderer is not running")
      qpos = data.qpos.copy()
      mocap_pos = data.mocap_pos.copy()
      mocap_quat = data.mocap_quat.copy()
      for array in (qpos, mocap_pos, mocap_quat):
        array.setflags(write=False)
      self._latest = _FrameSnapshot(qpos, float(data.time), mocap_pos, mocap_quat)
      self._condition.notify()

  def stop(self, timeout: float = 10.0) -> None:
    """Stop submission and join; OpenGL cleanup remains on the worker."""
    if not math.isfinite(timeout) or timeout < 0.0:
      raise ValueError("front camera shutdown timeout must be finite and nonnegative")
    with self._condition:
      self._stopping = True
      self._latest = None
      self._condition.notify_all()
      thread = self._thread
    if thread is None or thread is threading.current_thread():
      return
    thread.join(timeout)
    if thread.is_alive():
      raise TimeoutError("front camera rendering worker did not stop in time")

  def _record_error(self, error: BaseException) -> None:
    with self._condition:
      if self._error is None:
        self._error = error
      self._stopping = True
      self._latest = None
      self._condition.notify_all()

  def _run(self) -> None:
    renderer = None
    try:
      # Nothing about the live physics MjData is retained by this worker.
      data = mujoco.MjData(self._model)
      renderer = mujoco.Renderer(
        self._model, height=self._config.height, width=self._config.width,
      )
      self._ready.set()
      while True:
        with self._condition:
          self._condition.wait_for(lambda: self._stopping or self._latest is not None)
          if self._stopping:
            break
          frame = self._latest
          self._latest = None
        data.qpos[:] = frame.qpos
        data.time = frame.sim_time
        data.mocap_pos[:] = frame.mocap_pos
        data.mocap_quat[:] = frame.mocap_quat
        mujoco.mj_forward(self._model, data)
        renderer.update_scene(data, camera=CAMERA_NAME)
        rgb = renderer.render()
        with self._condition:
          stopping = self._stopping
        if not stopping:
          self._callback(rgb, frame.sim_time)
    except BaseException as exc:
      self._record_error(exc)
    finally:
      if renderer is not None:
        try:
          renderer.close()
        except BaseException as exc:
          self._record_error(exc)
      # Failed initialization must also wake start() immediately.
      self._ready.set()
