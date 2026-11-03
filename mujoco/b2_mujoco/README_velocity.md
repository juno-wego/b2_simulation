# Simulated velocity actuator

`b2_sim.launch.py` enables measured body-velocity feedback from `/b2/odom_gt`
inside the joint-policy runner. Nav2's chassis command remains unchanged. The
policy input compensates tracking error using bounded PI feedback (P=0.7,
I=0.5/s, measurement filter 0.25s). Motion still comes only from joint torques,
contacts and the shipped ONNX policy; pose and odometry are never overridden.

The exported policy was trained to stand below command norm 0.1. Feedback mode
activates gait for requests outside the physical SDK bridge's per-axis 0.01
deadband. It does not impose a minimum chassis speed. Correction is bounded to
0.35 per axis, integral to 0.25, and policy inputs to the trained velocity range.

Zero/expired commands and missing, invalid or stale velocity feedback clear
the correction and select the standing policy. Feedback requires `base_link`
body twist, strictly increasing timestamps, age <=0.25s and at most 0.05s clock
skew. Original requests expire after 0.5s. `/b2/policy_command` reports the
actual policy input for diagnosis; `/cmd_vel` remains the requested velocity.

The policy runner executable alone defaults to `velocity_feedback_topic: ""`
and preserves the original policy behavior. The MuJoCo launch explicitly enables
feedback; this is not a validated controller for the physical robot.

## Verification

With ROS sourced and the simulation virtualenv active:

```bash
python -m pytest -q test/test_velocity_servo.py test/test_velocity_feedback.py \
  test/test_velocity_tracking_physics.py
```

The physics regression runs the shipped scene, policy, motor gains and torque
limits. It measures forward/reverse/lateral/yaw and mixed requests, including
0.02 m/s motion and a subsequent zero command. It requires mean tracking error
below 0.009 m/s or rad/s, final stopped speed below 0.006, base height above
0.45m and tilt below 7 degrees. These flat-floor checks do not establish stair,
payload or real-robot fidelity. Full robot Nav2/BT acceptance is separate.
