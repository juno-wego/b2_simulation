"""Bring up the MuJoCo B2 and its velocity policy, nothing else.

Use this to check that the robot stands and walks before layering SLAM on top:

    ros2 launch b2_mujoco b2_sim.launch.py
    ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}'

`payload` selects which robot.  It sets the scene and the policy *together* on
purpose: b2_velocity.onnx was trained on an 83.5 kg robot and
b2_arm_velocity.onnx on a 105 kg one, and pairing the wrong two hands the policy
a centre of mass it has never seen.  Override `scene_file` / `policy_file`
individually only if you mean to.

    ros2 launch b2_mujoco b2_sim.launch.py                # bare B2
    ros2 launch b2_mujoco b2_sim.launch.py payload:=fr3   # B2 + FAIRINO FR3
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("b2_mujoco")
    use_sim_time = {"use_sim_time": True}

    # "" for the bare robot, "_arm" for the FR3 build.  Both the scene and the
    # policy filename are built from it, so they cannot drift apart.
    variant = PythonExpression(
        ["'_arm' if '", LaunchConfiguration("payload"), "' == 'fr3' else ''"]
    )

    sim = Node(
        package="b2_mujoco",
        executable="b2_sim",
        name="b2_mujoco_sim",
        output="screen",
        parameters=[
            use_sim_time,
            {
                "scene_file": LaunchConfiguration("scene_file"),
                "viewer": LaunchConfiguration("viewer"),
                "publish_ground_truth_tf": LaunchConfiguration("ground_truth_tf"),
            },
        ],
    )

    policy = Node(
        package="b2_mujoco",
        executable="b2_policy_runner",
        name="b2_policy_runner",
        output="screen",
        parameters=[
            use_sim_time,
            {"policy_file": LaunchConfiguration("policy_file")},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "payload",
            default_value="none",
            choices=["none", "fr3"],
            description="What the robot is carrying.  'fr3' swaps in the B2 + "
                        "FAIRINO FR3 scene and the policy trained for it.",
        ),
        DeclareLaunchArgument(
            "scene_file",
            default_value=PathJoinSubstitution([
                pkg_share, "models", ["b2", variant, "_nav_scene.xml"],
            ]),
            description="MuJoCo scene to simulate.  Defaults follow `payload`.",
        ),
        DeclareLaunchArgument(
            "policy_file",
            default_value=PathJoinSubstitution([
                pkg_share, "policy", ["b2", variant, "_velocity.onnx"],
            ]),
            description="Trained B2 velocity policy (ONNX, exported by "
                        "unitree_rl_mjlab).  Defaults follow `payload`.",
        ),
        DeclareLaunchArgument(
            "viewer", default_value="true", description="Open the MuJoCo viewer window."
        ),
        DeclareLaunchArgument(
            "ground_truth_tf",
            default_value="true",
            description="Publish odom->base_link from simulator ground truth. "
                        "Turn this off when KISS-ICP owns that transform.",
        ),
        sim,
        policy,
    ])
