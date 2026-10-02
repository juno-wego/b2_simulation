# B2 simulated front camera

This is the camera input stage for the MuJoCo AprilTag test scene. It provides
an RGB image, matching `CameraInfo`, and camera TFs. Tag detection, docking
control, HMI integration, and Nav2 configuration are separate stages; enabling
this camera does not configure them.

The camera is optional and defaults to **off** in both `b2_sim.launch.py` and
`simulation_bringup/bringup.launch.py`. Set `front_camera:=true` to enable it.
The launch argument becomes the simulator's Boolean `front_camera_enabled`
parameter. Camera parameters are applied at startup; restart the simulator to
apply changed values.

| Output or setting | Default |
| --- | --- |
| Image topic | `/b2/front_camera/image_raw` (`sensor_msgs/msg/Image`, `rgb8`) |
| Intrinsics topic | `/b2/front_camera/camera_info` (`sensor_msgs/msg/CameraInfo`) |
| Camera link frame | `b2/front_camera_link` |
| Image / CameraInfo frame | `b2/front_camera_optical_frame` |
| Resolution | 640 × 480 pixels |
| Rate | 10 Hz of simulation time |
| Vertical field of view | 60 degrees |
| Camera position in `base_link` | `(0.39079, 0, -0.013433)` metres |

The link uses ROS body axes (X forward, Y left, Z up). The optical frame uses
X right, Y down, Z forward, with static transforms from `base_link` through the
camera link to the optical frame. `CameraInfo` describes the ideal simulated
pinhole camera, with zero distortion. These defaults are simulation geometry;
they are not calibration results for a physical B2 camera.

## Build and launch

Use the existing B2 MuJoCo environment with `mujoco` and `onnxruntime` installed.
Build and source the changed packages:

```bash
cd /home/wego/shalom_ws
source /opt/ros/jazzy/setup.bash
colcon build --base-paths src/shalom \
  --packages-select b2_mujoco simulation_bringup
source install/setup.bash
source src/shalom/robot/third_party/b2_simulation/mujoco/b2_mujoco/b2_env.sh
```

For the camera-only test stage, run the existing simulator and velocity policy
with the source test scene:

```bash
ros2 launch b2_mujoco b2_sim.launch.py front_camera:=true \
  scene_file:=/home/wego/shalom_ws/src/shalom/robot/third_party/b2_simulation/mujoco/b2_mujoco/models/b2_tag_test_scene.xml
```

The same flag is forwarded by the full simulation stack:

```bash
ros2 launch simulation_bringup bringup.launch.py front_camera:=true \
  scene_file:=/home/wego/shalom_ws/src/shalom/robot/third_party/b2_simulation/mujoco/b2_mujoco/models/b2_tag_test_scene.xml
```

The tag test scene and texture are also installed under
`share/b2_mujoco/models`. The existing texture directory is literally ` tags`
(a leading space), and the package installation preserves that name so the
scene's relative texture path works. For an installed scene, use
`scene_file:="$(ros2 pkg prefix b2_mujoco)/share/b2_mujoco/models/b2_tag_test_scene.xml"`.

Rendering uses an offscreen MuJoCo renderer even when `viewer:=false`; it still
needs a working OpenGL backend. Use the display backend already configured for
the simulator, or select a supported headless backend such as `MUJOCO_GL=egl`
before launch if the machine has it available.

If the renderer fails while running, the simulator logs the error and stops
publishing camera images while physics and control continue.

## Inspect the camera

In another sourced terminal, show the image with `image_view` if that package is
installed:

```bash
ros2 run image_view image_view --ros-args \
  -r image:=/b2/front_camera/image_raw
```

Jazzy's `image_view` subscribes with sensor-data QoS by default
([upstream implementation](https://github.com/ros-perception/image_pipeline/blob/jazzy/image_view/src/image_view_node.cpp)).

Inspect the stream and optical transform:

```bash
ros2 topic hz /b2/front_camera/image_raw
ros2 topic echo /b2/front_camera/image_raw --field header --once \
  --qos-reliability best_effort
ros2 topic echo /b2/front_camera/camera_info --once \
  --qos-reliability best_effort
ros2 run tf2_ros tf2_echo base_link b2/front_camera_optical_frame
```

An image and its `CameraInfo` must have the same simulation timestamp and
`b2/front_camera_optical_frame` frame ID. Consecutive one-shot commands may
capture different frames; compare a simultaneously received pair when checking
exact timestamp equality. Check that the image is 640 × 480 `rgb8`, that
`CameraInfo` has the same dimensions and positive focal lengths, and that TF
connects `base_link` to the optical frame. The test panel should be visible
ahead of the spawn; its pose is a temporary visual test placement in MuJoCo
world coordinates, not a calibrated ROS map docking location.

With the default `front_camera:=false`, the simulator does not create camera
publishers or its camera renderer. A disabled camera therefore adds no image
rendering work to the existing LiDAR and control loop.
