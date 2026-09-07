# Environment for the B2 MuJoCo stack.  Source it, do not execute it:
#
#     source ~/shalom_ws/src/b2_simulation/mujoco/b2_mujoco/b2_env.sh
#
# The nodes need mujoco and onnxruntime.  Those live in a dedicated virtualenv so
# they stay out of the apt-managed system python; putting the venv's site-packages
# on PYTHONPATH lets the system interpreter (the one colcon builds node scripts
# against) import them while numpy, rclpy and the rest still come from ROS.

B2_VENV="${B2_VENV:-$HOME/shalom_ws/.venv-b2sim}"
B2_WS="${B2_WS:-$HOME/shalom_ws}"

source /opt/ros/jazzy/setup.bash
source "$B2_WS/install/setup.bash"

export PYTHONPATH="$B2_VENV/lib/python3.12/site-packages:$PYTHONPATH"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# b2_driver's cyclonedds_mujoco.xml / cyclonedds_client.xml pin <ParticipantIndex>,
# which only works when a single DDS participant exists -- they exist so the native
# unitree_mujoco bridge and the SDK client can find each other without colliding
# with a ROS domain-0 session.  This stack is pure ROS 2 and launches ~19 nodes, so
# a pinned index makes every node after the first die with
#   rtps_init: failed to create unicast sockets for domain 0 participant index N
# Clear it and let Cyclone assign indices normally.
if [ -n "${CYCLONEDDS_URI:-}" ] && grep -q "<ParticipantIndex>" "${CYCLONEDDS_URI#file://}" 2>/dev/null; then
  echo "b2_mujoco: clearing CYCLONEDDS_URI=$CYCLONEDDS_URI"
  echo "           (it pins a DDS participant index, which a multi-node launch cannot share)"
  unset CYCLONEDDS_URI
fi

python3 - <<'PY' || echo "  create it with: python3 -m venv --system-site-packages \$B2_VENV && \$B2_VENV/bin/pip install mujoco onnxruntime"
try:
    import mujoco, onnxruntime
    print(f"b2_mujoco env ready: mujoco {mujoco.__version__}, onnxruntime {onnxruntime.__version__}")
except ImportError as exc:
    raise SystemExit(f"b2_mujoco env INCOMPLETE: {exc}")
PY
