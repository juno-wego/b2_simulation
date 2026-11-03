import numpy as np
import pytest

from b2_mujoco.velocity_servo import VelocityServo


def servo():
    return VelocityServo([-1., -.8, -1.], [1.5, .8, 1.])


@pytest.mark.parametrize("desired,measurement", [
    ([0., 0., 0.], [.1, .1, .1]),
    ([.009, .009, .009], [0., 0., 0.]),
    ([.05, 0., 0.], None),
    ([float("nan"), 0., 0.], [0., 0., 0.]),
    ([.05, 0., 0.], [float("nan"), 0., 0.]),
])
def test_stop_or_invalid_input_discards_accumulated_correction(desired, measurement):
    tracker = servo()
    for _ in range(100):
        tracker.step([.05, 0., 0.], [0., 0., 0.], .02)
    assert tracker.integral[0] > 0.
    drive, walking = tracker.step(desired, measurement, .02)
    assert not walking
    assert drive == pytest.approx([0., 0., 0.])
    assert tracker.integral == pytest.approx([0., 0., 0.])


def test_feedback_tracks_the_requested_speed_without_a_minimum_velocity_floor():
    tracker = servo()
    drive, walking = tracker.step([.02, 0., 0.], [.02, 0., 0.], .02)
    assert walking
    assert drive == pytest.approx([.02, 0., 0.])


def test_saturation_stays_inside_training_limits_and_does_not_wind_up():
    tracker = servo()
    for _ in range(1000):
        drive, walking = tracker.step([1.5, .8, -1.], [0., 0., 0.], .02)
    assert walking
    assert drive == pytest.approx([1.5, .8, -1.])
    assert tracker.integral == pytest.approx([0., 0., 0.])
