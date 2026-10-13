"""Check the loaded terrain surface, native contacts and simulated LiDAR."""

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from b2_mujoco.lidar import RaycastLidar


MODELS = Path(__file__).resolve().parents[1] / "models"
WORLD_FILE = MODELS / "terrain_course.xml"
WORLD_MASK = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


@pytest.fixture(scope="module")
def world():
    model = mujoco.MjModel.from_xml_path(str(WORLD_FILE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def surface_height(world, x, y):
    model, data = world
    hit = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(
        model, data, np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]),
        WORLD_MASK, True, -1, hit,
    )
    assert distance >= 0.0
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(hit[0])) == "course_ground"
    return 5.0 - distance


@pytest.mark.parametrize("axis", ["x", "y"])
def test_round_profile_merges_with_flat_ground_and_has_no_crest_corner(world, axis):
    # Sample the actual collision/raycast surface, including flat ground at both
    # ends. A pair of straight ramps would have an abrupt slope reversal here.
    if axis == "x":
        coordinates = np.linspace(-10.2, 4.2, 145)
        heights = np.array([surface_height(world, x, 0.0) for x in coordinates])
        maximum_slope = 9.0
    else:
        coordinates = np.linspace(-4.6, 4.6, 93)
        heights = np.array([surface_height(world, -3.0, y) for y in coordinates])
        maximum_slope = 14.0

    middle = len(heights) // 2
    np.testing.assert_allclose(heights, heights[::-1], atol=1e-6)
    np.testing.assert_allclose(heights[[0, -1]], 0.0, atol=1e-8)
    assert heights[middle] == pytest.approx(0.6, abs=1e-6)
    assert np.all(np.diff(heights[:middle + 1]) >= -1e-8)
    assert np.all(np.diff(heights[middle:]) <= 1e-8)

    slopes = np.degrees(np.arctan(np.diff(heights) / np.diff(coordinates)))
    assert np.max(np.abs(slopes)) < maximum_slope
    # The 0.2 m sampling introduces small normal changes, below 2.3 degrees,
    # rather than the previous pair of ramps' 30-degree crest corner.
    assert np.max(np.abs(np.diff(slopes))) < 2.3
    assert np.max(np.abs(slopes[[0, -1, middle - 1, middle]])) < 1.2


@pytest.mark.parametrize("x,y", [(-12.0, 0.0), (-7.0, 0.0), (-3.0, 0.0), (-3.0, 2.0)])
def test_foot_sized_sphere_contacts_flat_and_curved_ground(world, x, y):
    height = surface_height(world, x, y)
    # The B2's foot collision spheres have radius 0.032 m. Give the probe a small
    # penetration to confirm that the rendered hill is also a physical surface.
    model = mujoco.MjModel.from_xml_string(f"""
        <mujoco>
          <include file="{WORLD_FILE}"/>
          <worldbody>
            <body name="probe" pos="{x} {y} {height + 0.030}">
              <freejoint/>
              <geom name="probe_foot" type="sphere" size="0.032" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "course_ground")
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe_foot")
    contacts = [contact for contact in data.contact
                if set(contact.geom) == {ground_id, foot_id}]
    assert contacts
    assert -0.005 < min(contact.dist for contact in contacts) < 0.0
    assert all(abs(contact.frame[2]) > 0.95 for contact in contacts)


def test_simulated_lidar_sees_the_rounded_hill(world):
    model, data = world
    lidar = RaycastLidar(num_rings=1, num_azimuth=1,
                         fov_lower_deg=-90.0, fov_upper_deg=-90.0,
                         noise_std=0.0)
    points = lidar.scan(model, data, np.array([-3.0, 0.0, 2.0]), np.eye(3))
    assert points.shape == (1, 3)
    assert 2.0 + points[0, 2] == pytest.approx(0.6, abs=1e-6)


def test_b2_starts_clear_of_the_hill_on_flat_ground():
    model = mujoco.MjModel.from_xml_path(str(MODELS / "b2_terrain_course_scene.xml"))
    data = mujoco.MjData(model)
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    assert stand >= 0
    mujoco.mj_resetDataKeyframe(model, data, stand)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qpos[:2], [-12.0, 0.0])
    assert surface_height((model, data), -12.0, 0.0) == pytest.approx(0.0, abs=1e-8)
    assert not data.ncon or min(contact.dist for contact in data.contact) > -0.005
