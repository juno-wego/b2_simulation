"""Raycast LiDAR for the MuJoCo B2 scene.

Models a spinning multi-beam scanner: one optical centre, a fixed ring/azimuth
grid of directions, and `mujoco.mj_multiRay` for the intersection test.  Rays are
restricted to geom group 0 (the world), which is how the scan avoids returning
hits on B2's own legs without per-body exclusion.
"""

from __future__ import annotations

import numpy as np

import mujoco

# The world scene puts every static geom in group 0; the robot's own geoms are in
# groups 1-3, so this mask makes the LiDAR blind to the robot itself.
_WORLD_GROUP_MASK = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


class RaycastLidar:
  """Fixed-pattern multi-beam LiDAR sampled by batched MuJoCo raycasts."""

  def __init__(
    self,
    num_rings: int = 32,
    num_azimuth: int = 720,
    fov_lower_deg: float = -30.0,
    fov_upper_deg: float = 12.0,
    range_min: float = 0.4,
    range_max: float = 30.0,
    noise_std: float = 0.01,
  ) -> None:
    self.range_min = float(range_min)
    self.range_max = float(range_max)
    self.noise_std = float(noise_std)

    elev = np.deg2rad(np.linspace(fov_lower_deg, fov_upper_deg, num_rings))
    azim = np.linspace(-np.pi, np.pi, num_azimuth, endpoint=False)
    el, az = np.meshgrid(elev, azim, indexing="ij")

    # Directions in the sensor frame, flattened ring-major (ring 0 first).
    self._dirs_local = np.stack(
      (np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), axis=-1
    ).reshape(-1, 3)

    self.num_rays = self._dirs_local.shape[0]
    self._dirs_world = np.zeros((self.num_rays, 3), dtype=np.float64)
    self._dist = np.zeros(self.num_rays, dtype=np.float64)
    self._geomid = np.zeros(self.num_rays, dtype=np.int32)
    self._rng = np.random.default_rng(0)

  def scan(
    self,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    origin: np.ndarray,
    rotation: np.ndarray,
  ) -> np.ndarray:
    """Return hit points as (N, 3) in the sensor frame.

    Args:
      origin: sensor optical centre in world coordinates, shape (3,).
      rotation: sensor->world rotation matrix, shape (3, 3).
    """
    # Rays are cast in world coordinates; (R @ v.T).T == v @ R.T.
    np.matmul(self._dirs_local, rotation.T, out=self._dirs_world)

    mujoco.mj_multiRay(
      m=model,
      d=data,
      pnt=np.ascontiguousarray(origin, dtype=np.float64),
      vec=self._dirs_world.reshape(-1),
      geomgroup=_WORLD_GROUP_MASK,
      flg_static=True,
      bodyexclude=-1,
      geomid=self._geomid,
      dist=self._dist,
      normal=None,
      nray=self.num_rays,
      cutoff=self.range_max,
    )

    # mj_multiRay reports -1 for rays that hit nothing.
    hit = (self._geomid >= 0) & (self._dist >= self.range_min) & (self._dist <= self.range_max)
    ranges = self._dist[hit]
    if self.noise_std > 0.0:
      ranges = ranges + self._rng.normal(0.0, self.noise_std, ranges.shape)

    # Back to the sensor frame: the local direction times the measured range.
    return (self._dirs_local[hit] * ranges[:, None]).astype(np.float32)
