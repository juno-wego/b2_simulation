"""Low-speed gait must preserve the requested velocity and keep zero stopped."""
from types import SimpleNamespace

import numpy as np
import pytest

from b2_mujoco.policy_runner_node import B2PolicyRunner


@pytest.mark.parametrize("threshold,cmd,walking", [
    (.1, [.06, 0., 0.], False),
    (.02, [.06, 0., 0.], True),
    (.02, [0., .04, 0.], True),
    (.02, [0., 0., .05], True),
    (.02, [0., 0., 0.], False),
    (.02, [.001, 0., 0.], False),
])
def test_gait_phase_responds_to_small_commands_without_changing_velocity(threshold, cmd, walking):
    observations = []

    def run(outputs, inputs):
        observations.append(inputs["obs"].copy())
        return [np.zeros((1, 12), dtype=np.float32)]

    runner = SimpleNamespace(
        get_parameter=lambda _: SimpleNamespace(value=threshold),
        _phase_step=7, dt=.02, gait_period=.7,
        sdk_to_policy=np.arange(12), policy_to_sdk=np.arange(12),
        default_q_policy=np.zeros(12), _last_action=np.zeros(12),
        obs_dim=47, input_name="obs", session=SimpleNamespace(run=run),
        action_scale=.25, kp_sdk=np.ones(12), kd_sdk=np.ones(12),
        _publish_cmd=lambda *args: None,
    )
    state = SimpleNamespace(imu_state=SimpleNamespace(gyroscope=[0., 0., 0.], quaternion=[1., 0., 0., 0.]))
    B2PolicyRunner._policy_step(runner, state, np.zeros(12), np.zeros(12), np.asarray(cmd))
    obs = observations[0][0]
    assert obs[6:9] == pytest.approx(cmd), "requested velocity must never be scaled up"
    assert np.linalg.norm(obs[9:11]) == pytest.approx(1. if walking else 0.)
