# b2_simulation

Unitree B2 시뮬레이션. 목적이 두 가지다.

**1. 실기 대체.** 센서 데이터가 나가고 주행 명령이 들어가면 걷는다. 실기와 **같은
메시지**(`unitree_go/LowState`, `LowCmd`)를 주고받으므로, 상위 제어 입장에서 실기와
구별되지 않는다. 그래서 SLAM도 Nav2도 여기 없다 — 그건 상위인 `shalom`의 일이다.

**2. 정책 학습.** 실기에 올릴 보행 정책을 여기서 학습한다. 학습과 배포가 같은 MuJoCo
물리를 쓰므로 sim-to-sim 격차가 없다.

| 디렉터리 | 시뮬레이터 | 상태 |
|---|---|---|
| [`mujoco/`](mujoco/) | MuJoCo | **동작함** — 실기 대체 + RL 학습 |

## 인터페이스

실기와 동일하다. 이게 이 저장소의 핵심이다.

| 방향 | 토픽 | 타입 |
|---|---|---|
| 나감 | `/lowstate` | `unitree_go/LowState` — 관절, IMU, 발 접촉 |
| 나감 | `/b2/points` | `sensor_msgs/PointCloud2` — 32빔 LiDAR |
| 나감 | `/b2/imu`, `/b2/joint_states`, `/clock` | |
| 들어옴 | `/lowcmd` | `unitree_go/LowCmd` — 관절 명령 |
| 들어옴 | `/cmd_vel` | `geometry_msgs/Twist` — 주행 명령 |

## 실행

```bash
source ~/shalom_ws/src/b2_simulation/mujoco/b2_mujoco/b2_env.sh
ros2 launch b2_mujoco b2_sim.launch.py
```

로봇이 damping → 웅크림 → 기립 순서로 일어선 뒤 정책에 제어권을 넘긴다. 그 뒤
`/cmd_vel`을 주면 걷는다. 자율주행 전체는 `shalom`에서 띄운다.

자세한 내용은 [`mujoco/README.md`](mujoco/README.md).

## 이웃 저장소

| 저장소 | 역할 |
|---|---|
| `shalom` | 자율주행 상위 제어 — 이 시뮬레이터나 실기를 골라 붙인다 |
| `nav2_3d` | 3D 지면분할 인식 파이프라인 |
| `b2_driver` | 실기 드라이버. 이 저장소가 대체하려는 대상 |
