"""The ROS boundary must stop the servo when feedback or requests expire."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from nav_msgs.msg import Odometry

from b2_mujoco.policy_runner_node import B2PolicyRunner
from b2_mujoco.velocity_servo import VelocityServo


def runner():
    parameters = {"velocity_feedback_frame": "base_link", "velocity_feedback_timeout": .25}
    return SimpleNamespace(
        _lock=threading.Lock(), _now=lambda: 100.1,
        _velocity=None, _velocity_stamp=0.,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        _servo=VelocityServo([-1., -.8, -1.], [1.5, .8, 1.]),
        _state=SimpleNamespace(motor_state=[SimpleNamespace(q=0., dq=0.)] * 12),
        _cmd=np.array([.02, 0., 0.]), _cmd_stamp=100., cmd_timeout=.5,
        _mode="policy", _mode_t=0., dt=.02,
        _publish_policy_input=lambda cmd: None,
    )


def sample(stamp=100., frame="base_link", speed=0.):
    message = Odometry()
    message.header.stamp.sec = int(stamp)
    message.header.stamp.nanosec = round((stamp % 1.) * 1e9)
    message.child_frame_id = frame
    message.twist.twist.linear.x = speed
    return message


@pytest.mark.parametrize("message", [
    sample(99.), sample(101.), sample(frame="odom"), sample(speed=float("nan")),
])
def test_invalid_feedback_stops_instead_of_using_old_correction(message):
    node = runner()
    B2PolicyRunner._on_velocity(node, sample())
    assert node._velocity is not None
    B2PolicyRunner._on_velocity(node, message)
    assert node._velocity is None
    output = []
    node._policy_step = lambda *args: output.append(args)
    B2PolicyRunner._control_step(node)
    assert output[0][-2] == pytest.approx([0., 0., 0.])
    assert output[0][-1] is False


@pytest.mark.parametrize("expired", ["velocity", "request"])
def test_loss_of_stream_stops_and_clears_integral(expired):
    node = runner()
    B2PolicyRunner._on_velocity(node, sample())
    output = []
    node._policy_step = lambda *args: output.append(args)
    B2PolicyRunner._control_step(node)
    assert output[-1][-1] is True
    if expired == "velocity":
        node._velocity_stamp = 99.
    else:
        node._cmd_stamp = 99.
    B2PolicyRunner._control_step(node)
    assert output[-1][-2] == pytest.approx([0., 0., 0.])
    assert output[-1][-1] is False
    assert node._servo.integral == pytest.approx([0., 0., 0.])


def test_replayed_sample_does_not_replace_current_measurement():
    node = runner()
    B2PolicyRunner._on_velocity(node, sample(speed=.02))
    B2PolicyRunner._on_velocity(node, sample(speed=1.))
    assert node._velocity == pytest.approx([.02, 0., 0.])
