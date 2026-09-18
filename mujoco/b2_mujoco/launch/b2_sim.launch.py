"""Bring up the MuJoCo B2 and its velocity policy, nothing else.

Use this to check that the robot stands and walks before layering SLAM on top:

    ros2 launch b2_mujoco b2_sim.launch.py
    ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}'

This package simulates the bare B2 only.  `scene_file` and `policy_file` may
be overridden for B2 development, but must remain a matched B2 scene/policy.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("b2_mujoco")
    use_sim_time = {"use_sim_time": True}

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
            "scene_file",
            default_value=PathJoinSubstitution([
                pkg_share, "models", "b2_nav_scene.xml",
            ]),
            description="MuJoCo scene to simulate.",
        ),
        DeclareLaunchArgument(
            "policy_file",
            default_value=PathJoinSubstitution([
                pkg_share, "policy", "b2_velocity.onnx",
            ]),
            description="Trained B2 velocity policy (ONNX, exported by "
                        "unitree_rl_mjlab).",
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
