"""Run the shipped policy and joint torques without pose/velocity injection.

Requires the simulation virtualenv. This tests the flat-floor actuator;
full ROS navigation acceptance is a separate check.
"""
import math
from pathlib import Path

import numpy as np
import pytest

import b2_mujoco._venv  # noqa: F401
from b2_mujoco.policy_runner_node import _quat_to_projected_gravity
from b2_mujoco.velocity_servo import VelocityServo

mujoco = pytest.importorskip("mujoco")
ort = pytest.importorskip("onnxruntime")


def simulate(command, seconds=12.):
    root = Path(__file__).resolve().parents[1]
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(root / "policy/b2_velocity.onnx"),
                                   sess_options=options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    model = mujoco.MjModel.from_xml_path(str(root / "models/b2_nav_scene.xml"))
    data = mujoco.MjData(model)
    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, stand)
    mujoco.mj_forward(model, data)

    def metadata_array(name):
        return np.array([float(value) for value in metadata[name].split(",")])

    def sensor(name):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return slice(model.sensor_adr[sid], model.sensor_adr[sid] + model.sensor_dim[sid])

    joints = metadata["joint_names"].split(",")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joints]
    positions = np.array([model.jnt_qposadr[jid] for jid in joint_ids])
    velocities = np.array([model.jnt_dofadr[jid] for jid in joint_ids])
    actuators = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name[:-6])
                          for name in joints])
    torque_limits = np.abs(model.actuator_ctrlrange[actuators, 1])
    nominal = metadata_array("default_joint_pos")
    kp, kd = metadata_array("joint_stiffness"), metadata_array("joint_damping")
    scale = float(metadata["action_scale"])
    quat, gyro = sensor("imu_quat"), sensor("imu_gyro")
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    dt, control_dt = float(model.opt.timestep), .02
    decimation = round(control_dt / dt)
    phase_step, max_tilt = 0, 0.
    last_action = np.zeros(12, dtype=np.float32)
    target = nominal.copy()
    tracker = VelocityServo([-1., -.8, -1.], [1.5, .8, 1.])
    twist = np.zeros(6)
    records = []
    total = 4. + seconds + 3.

    for step in range(round(total / dt)):
        t = data.time
        if step % decimation == 0:
            # XBODY is the base_link frame; BODY uses rotated principal inertia axes.
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, base, twist, 1)
            measured = np.array([twist[3], twist[4], twist[2]])
            requested = command if 4. <= t < 4. + seconds else np.zeros(3)
            drive, walking = tracker.step(requested, measured, control_dt)
            if t < 1.:
                target = nominal.copy()
            else:
                theta = 2. * math.pi * ((phase_step * control_dt) % .7) / .7
                phase = [math.sin(theta), math.cos(theta)] if walking else [0., 0.]
                phase_step += 1
                observation = np.concatenate((
                    data.sensordata[gyro], _quat_to_projected_gravity(data.sensordata[quat]),
                    drive, phase, data.qpos[positions] - nominal,
                    data.qvel[velocities], last_action,
                )).astype(np.float32)
                last_action = session.run(None, {session.get_inputs()[0].name:
                                                observation[None, :]})[0][0]
                target = nominal + scale * last_action
            q = data.qpos[3:7]
            tilt = math.acos(np.clip(1. - 2. * (q[1] ** 2 + q[2] ** 2), -1., 1.))
            max_tilt = max(max_tilt, tilt)
            records.append((t, measured.copy(), data.qpos[2]))
            assert data.qpos[2] > .25, "robot fell over"
        gains = (400., 8.) if t < 1. else (kp, kd)
        torque = gains[0] * (target - data.qpos[positions]) - gains[1] * data.qvel[velocities]
        data.ctrl[actuators] = np.clip(torque, -torque_limits, torque_limits)
        mujoco.mj_step(model, data)

    steady = [r for r in records if 4. + seconds / 2. <= r[0] < 4. + seconds]
    stopped = [r for r in records if r[0] >= total - 1.]
    return {
        "command": command,
        "mean_v": np.mean([r[1] for r in steady], axis=0).tolist(),
        "stop_mean_v": np.mean([r[1] for r in stopped], axis=0).tolist(),
        "max_tilt_deg": math.degrees(max_tilt),
        "mean_height": float(np.mean([r[2] for r in steady])),
    }


@pytest.mark.parametrize("command", [
    [.02, 0., 0.], [-.03, 0., 0.], [0., .03, 0.],
    [0., 0., .05], [0., 0., -.07], [.05, .02, -.07], [.3, 0., 0.],
])
def test_policy_physically_tracks_low_velocities_and_stops(command):
    result = simulate(command)
    assert np.max(np.abs(np.asarray(result["mean_v"]) - command)) < .009, result
    assert np.max(np.abs(result["stop_mean_v"])) < .006, result
    assert result["max_tilt_deg"] < 7., result
    assert result["mean_height"] > .45, result
