from glob import glob

from setuptools import find_packages, setup

package_name = "b2_mujoco"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/docs", ["README_camera.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/models", glob("models/*.xml")),
        ("share/" + package_name + "/models/assets", glob("models/assets/*")),
        # MJCF texture references preserve the existing leading-space directory.
        ("share/" + package_name + "/models/ tags", glob("models/ tags/*.png")),
        ("share/" + package_name + "/policy", glob("policy/*.onnx")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="juno",
    maintainer_email="junoyoo@wego-robotics.com",
    description="MuJoCo B2 simulation with an RL velocity policy for Nav2 SLAM.",
    license="Apache-2.0",
    # Plain scripts rather than console_scripts: these keep a `#!/usr/bin/env
    # python3` shebang, so they run under whichever interpreter is on PATH.  The
    # nodes need mujoco and onnxruntime, which live in a virtualenv, not in the
    # system python that colcon itself runs under.
    scripts=[
        "scripts/b2_sim",
        "scripts/b2_policy_runner",
        "scripts/b2_train_progress",
    ],
)
