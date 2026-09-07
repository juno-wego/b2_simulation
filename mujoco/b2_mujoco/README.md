# b2_mujoco

MuJoCo 시뮬레이션 위에서 Unitree B2를 **실제 로봇과 동일한 저수준 인터페이스로**
걷게 하고, 그 위에 3D LiDAR 지면 분할 → SLAM → Nav2 파이프라인을 얹는 패키지다.

핵심은 두 노드다.

| 노드 | 입력 | 출력 |
|---|---|---|
| `b2_sim` | `unitree_go/LowCmd` (`/lowcmd`) | `unitree_go/LowState` (`/lowstate`), `/b2/points`, `/b2/imu`, `/b2/joint_states`, `/b2/odom_gt`, `/clock` |
| `b2_policy_runner` | `/lowstate` + `/cmd_vel` | `/lowcmd` |

시뮬레이터와 실기가 **같은 메시지**를 주고받으므로, `b2_policy_runner`는 코드 변경
없이 실제 B2에도 붙는다. DDS 상대만 바뀐다.

## 왜 정책을 새로 학습했나

B2용 velocity policy는 공개된 것이 없다. `unitree_rl_lab`의 B2 배포 설정은
`logs/rsl_rl/unitree_b2_velocity`를 참조하지만 저장소에 가중치가 없고, Isaac Sim이
필요하다. 대신 **`unitree_rl_mjlab`에 B2를 추가해서 직접 학습**했다. 이 선택의
이유는 하나다: mjlab은 MuJoCo로 학습하고 배포 대상도 MuJoCo이므로 **sim-to-sim
격차가 0**이다.

학습 자산은 이 저장소가 아니라 `~/unitree_rl_mjlab`에 있다.

```
src/assets/robots/unitree_b2/           # b2_constants.py + mjlab 규약에 맞춘 MJCF
src/tasks/velocity/config/b2/           # Unitree-B2-Flat / Unitree-B2-Rough 태스크
```

MJCF는 Unitree 공식 `unitree_mujoco/unitree_robots/b2/b2.xml`에서 생성했다.
mjlab이 요구하는 것만 덧붙였다: 이름 붙은 `*_collision` geom, 발 site(FL/FR/RL/RR),
mjlab IMU 센서 이름. 물리 파라미터는 손대지 않았다.

액추에이터 한계와 기준 자세는 Unitree 자신의 B2 배포 설정에서 가져왔다.

- hip/thigh 200 Nm, calf 320 Nm (M107 구동계)
- kp 160, kd 5
- 기준 자세 `(hip, thigh, calf) = (0, 0.8, -1.5)`
  — B2의 `FixStand` FSM이 정책에 제어권을 넘기기 직전에 유지하는 바로 그 자세다.
  덕분에 정책의 첫 action(=0)이 무동작이 되어 인계가 매끄럽다.

Go2 대비 B2의 다리가 약 1.65배 길므로 보행 주기(0.6→0.7 s)와 발 들어올림
높이(0.10→0.15 m)만 스케일했다.

## 학습

```bash
cd ~/unitree_rl_mjlab
WANDB_MODE=offline .venv/bin/python scripts/train.py Unitree-B2-Flat \
  --env.scene.num-envs=4096 --agent.max-iterations=6000
```

RTX 4060 Laptop(8 GB)에서 4096 envs 기준 약 1.05 s/iteration. 100 iteration마다
`logs/rsl_rl/b2_velocity/<run>/policy.onnx`가 갱신된다. 학습이 끝나면 복사한다.

```bash
cp ~/unitree_rl_mjlab/logs/rsl_rl/b2_velocity/<run>/policy.onnx \
   ~/shalom_ws/src/b2_simulation/mujoco/b2_mujoco/policy/b2_velocity.onnx
```

진행 상황은 아래로 본다. `ros2 run`으로 실행해야 한다. ROS는 패키지 실행 파일을
`install/<pkg>/lib/<pkg>/`에만 설치하므로 이름만으로는 PATH에서 찾지 못한다.

```bash
ros2 run b2_mujoco b2_train_progress          # 5초마다 갱신
ros2 run b2_mujoco b2_train_progress --once   # 스냅샷 한 번
```

```
  [###############################.................]  66%   4017 / 6000
  elapsed 01:19:08   remaining (ETA) 00:39:02   iteration time 1.17s
  mean reward 47.11   episode length 1000.00 / 1000
  velocity tracking error   linear xy 0.405   angular yaw 0.477
  termination share         fell over 0.0000  illegal contact 0.0000
```

`train.py`는 진행 상황을 stdout으로만 내보내므로 로그 파일 위치는 실행할 때
리다이렉트한 곳이다. 뷰어는 관례적인 위치들 중 가장 최근 것을 쓰고, 필요하면
경로를 직접 넘긴다: `ros2 run b2_mujoco b2_train_progress /path/to/train.log`.

거친 지형용 정책이 필요하면 `Unitree-B2-Rough`를 쓴다. 다만 height scan 관측이
추가되므로 `b2_policy_runner`의 관측 조립도 함께 확장해야 한다(현재는 flat 전용,
관측 47차원).

`b2_policy_runner`는 관절 순서·기준 자세·PD 게인·action scale·관측 레이아웃을
**ONNX 메타데이터에서 읽는다.** 체크포인트와 런너가 어긋날 수 없고, 어긋나면
조용히 이상 동작하는 대신 기동 시 예외로 멈춘다.

## 환경 준비

노드는 `mujoco`와 `onnxruntime`이 필요하다. apt가 관리하는 시스템 python을
건드리지 않도록 전용 venv에 넣고, PYTHONPATH로만 얹는다.

```bash
python3 -m venv --system-site-packages ~/shalom_ws/.venv-b2sim
~/shalom_ws/.venv-b2sim/bin/pip install mujoco onnxruntime

cd ~/shalom_ws
colcon build --symlink-install --packages-select b2_mujoco
```

이후 매 터미널에서:

```bash
source ~/shalom_ws/src/b2_simulation/mujoco/b2_mujoco/b2_env.sh
```

`b2_env.sh`가 ROS, 워크스페이스, venv site-packages, CycloneDDS를 한 번에 잡는다.

### CYCLONEDDS_URI 주의

`b2_driver/README.md`의 네이티브 MuJoCo 검증 절차를 따라 했다면 셸에
`CYCLONEDDS_URI`가 남아 있을 수 있다. 그 설정 파일들은 `<ParticipantIndex>`를
**고정**한다. DDS 참가자가 하나일 때만 성립하는 값이라, 노드를 19개 띄우는 이
launch에서는 첫 노드만 인덱스를 잡고 나머지가 전부 죽는다.

```
rtps_init: failed to create unicast sockets for domain 0 participant index 20
rmw_create_node: failed to create domain, error Error
```

`b2_env.sh`는 이 변수를 자동으로 지우고 알려준다. 같은 터미널에서 네이티브
`unitree_mujoco` 검증을 다시 하려면 그때 `CYCLONEDDS_URI`를 새로 export 한다.

## 실행

### 1. 걷기만 확인

```bash
ros2 launch b2_mujoco b2_sim.launch.py
```

MuJoCo 뷰어가 열리고 로봇이 damping → 웅크림 → 기립 순서로 일어선 뒤 정책에
제어권을 넘긴다. 다른 터미널에서:

```bash
source ~/shalom_ws/src/b2_simulation/mujoco/b2_mujoco/b2_env.sh
ros2 topic pub -r 50 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.8}}'
```

`shalom`의 GUI 조종기도 그대로 쓸 수 있다.

```bash
ros2 run shalom control_gui
```

기동 FSM은 실기와 같다. 수동 전환은 다음과 같다.

```bash
ros2 topic pub --once /b2/mode std_msgs/msg/String '{data: damping}'   # 힘 빼기
ros2 topic pub --once /b2/mode std_msgs/msg/String '{data: stand}'     # 기립
ros2 topic pub --once /b2/mode std_msgs/msg/String '{data: policy}'    # 정책
```

현재 상태는 `/b2/mode_state`(latched)에서 읽는다. 자동 진행을 끄려면
`-p auto_start:=false`.

### 2. SLAM + Nav2 전체

```bash
ros2 launch b2_mujoco b2_slam.launch.py
```

파이프라인:

```
b2_sim /b2/points ─┬─ ground_segmentation ─ ground_filter ─ pointcloud_to_laserscan ─ slam_toolbox ─ /map
                   └─ kiss_icp ─ odom → base_link
Nav2 goal ─ /cmd_vel ─ b2_policy_runner ─ /lowcmd ─ b2_sim
```

RViz에서 `Nav2 Goal`로 목표를 찍으면 Nav2가 `/cmd_vel`을 내고, 정책이 그걸 걸음으로
바꾼다. 지도가 없는 상태에서 시작하므로 **이미 관측된 빈 공간**을 목표로 준다.

`ground_truth_tf`는 이 스택에서 자동으로 꺼진다. `odom → base_link`는 KISS-ICP가
소유하며, 한 변에 publisher가 둘이면 TF 트리가 깨진다.

## 검증 결과

아래는 학습 1300 iteration 시점의 중간 정책으로 측정한 값이다. 최종 정책은 이보다
낫다.

**보행**

| 항목 | 결과 |
|---|---|
| 전진 `x=0.8` 10초 | 6.23 m 이동 (0.62 m/s, 명령 대비 78%) |
| 10초 동안 횡방향 이탈 | −0.43 m |
| 몸통 높이 | 0.53 m 유지 (넘어지지 않음) |
| 사각 루프 약 25 m | 네 구간 모두 z ≈ 0.538 유지 |

**SLAM 파이프라인** (모든 단계 약 9.5 Hz)

```
/b2/points 9.5 → ground_points 9.3 / obstacle_points 9.6
           → slam_points 8.7 → obstacle_scan 9.5 → /map
/kiss/odometry 9.6 → odom → base_link
```

**KISS-ICP 정확도** (ground truth 대비)

| 이동 거리 | 오차 |
|---|---|
| 약 7.8 m | 1.5 cm |
| 약 25 m (사각 루프) | 2.1 cm |

**지도**: 239 × 179 셀 @ 0.1 m = 23.9 × 17.9 m. 씬의 24 × 18 m 홀과 일치하고,
칸막이 네 개·기둥·상자가 모두 제자리에 찍혔다. 미관측 셀 0개.

**Nav2 목표 주행** (`NavigateToPose`, map 프레임 (5.0, 0.0))

```
start   (-0.000,  0.001, 0.537)
result  SUCCEEDED
end     ( 5.072,  0.045, 0.535)     목표에서 7.2 cm
```

RViz 목표 → 플래너 → 컨트롤러 → `/cmd_vel` → 정책 → 관절 토크까지 전 구간이
연결돼 있고, 주행 내내 몸통 높이가 유지된다.

## 3D 분할과 2D 매핑의 역할 분담

인식 프론트엔드는 전부 3D이고, 매핑 백엔드만 2D다.

```
/b2/points (3D LiDAR)
   ├─ ground_segmentation (GSeg3D)   지면/장애물 분리 (셀 평면 피팅)
   │    ├─ ground_filter             지역 지면 기준 0.15~1.50 m 밴드
   │    │    └─ pointcloud_to_laserscan → slam_toolbox → /map   (2D)
   │    ├─ local_costmap  GroundConsistencyLayer                (3D 직접)
   │    └─ global_costmap obstacle_layer                        (3D 직접)
   └─ kiss_icp                       3D LiDAR odometry
```

지면은 3D로 판단해 지도에서 빼고, 벽·기둥·상자만 2D 격자에 남는다. Nav2는 SLAM
지도를 계획에 직접 쓰지 않는다. costmap은 실시간 장애물 포인트로 만들고,
slam_toolbox는 `map → odom` 보정과 사람이 볼 지도를 담당한다.

## 시뮬레이션 LiDAR

`mujoco.mj_multiRay`로 매 스캔을 일괄 레이캐스트한다. 기본값은 32링 × 720방위
(23,040 rays, 약 7.5 ms/scan), 수직 −30°~+12°, 최대 30 m, 10 Hz로
Velodyne HDL-32E급이다. 광학 중심은 B2 LiDAR 마운트 위치인 base_link 기준
`(0.34218, 0, 0.20)`이다.

레이는 **geom group 0(월드)만** 맞춘다. 그래서 body 단위 제외 없이도 로봇 자기
다리를 스캔하지 않는다. 씬에 물체를 추가할 때는 반드시 `class="world"`를 붙여야
LiDAR에 보인다.

주요 파라미터: `lidar_rings`, `lidar_azimuth`, `lidar_fov_lower_deg`,
`lidar_fov_upper_deg`, `lidar_range_max`, `lidar_rate_hz`, `lidar_offset`.

## 씬

씬 파일은 세 개다. 홀은 한 군데에만 있다.

```
models/nav_world.xml           24 x 18 m 실내 홀 + 솔버 설정 (로봇 없음)
models/b2_nav_scene.xml        b2.xml      + nav_world.xml
models/b2_arm_nav_scene.xml    b2_arm.xml  + nav_world.xml
```

SLAM이 루프 클로저를 걸 만한 구조(벽, 칸막이, 기둥, 상자)가 있어야 하기 때문에
빈 평면이 아니다. 두 로봇이 **같은 홀**을 돌아야 비교가 되므로 홀은 복제하지 않고
`nav_world.xml` 하나로 공유한다.

씬을 고칠 때 두 가지를 지켜야 한다.

- 월드 geom은 `class="world"`로 두어 **질량이 0**이 되게 한다. 월드에 용접된
  물체에 실제 밀도를 주면 `mjModel.stat.meaninertia`가 수십 배로 뛰고, 접촉
  솔버의 기준 스케일이 과도하게 뻣뻣해져 로봇이 바닥에서 튕겨 나간다.
- **원점 주변을 비워 둔다.** 로봇이 거기서 스폰된다. 기둥과 겹치면 스폰 즉시
  수십 개의 접촉이 생겨 로봇이 날아간다.

다른 씬을 쓰려면 `scene_file:=/abs/path.xml`. 씬에는 `stand` 키프레임이 있어야 한다.

## FR3 팔을 얹은 B2

등에 FAIRINO FR3 협동로봇(자중 15 kg, 가반 3 kg, 리치 622 mm)을 얹은 구성이다.
컨트롤 박스·마운트·DC-DC까지 **21.5 kg**, 총 105.0 kg. B2의 이동 정격 40 kg 안이다.

```bash
SHARE=$(ros2 pkg prefix b2_mujoco)/share/b2_mujoco
ros2 launch b2_mujoco b2_sim.launch.py \
    scene_file:=$SHARE/models/b2_arm_nav_scene.xml \
    policy_file:=$SHARE/policy/b2_arm_velocity.onnx
```

**정책과 모델은 반드시 짝을 맞춰야 한다.** `b2_velocity.onnx`는 83.5 kg 로봇용이고
`b2_arm_velocity.onnx`는 105 kg 로봇용이다. 서로 바꿔 넣으면 학습한 적 없는 무게중심을
만나게 된다.

`models/b2_arm.xml`은 손으로 쓴 파일이 아니라 `b2.xml`에서 생성된다. 학습 에셋을
만드는 것과 **같은 스크립트**다.

```bash
python3 ../rl_training/tools/make_b2_arm_asset.py \
    --deploy-in models/b2.xml --deploy-out models/b2_arm.xml
```

팔은 트렁크 중심보다 5 cm 뒤(x = −0.05)에 얹었다. 전방 LiDAR 마스트가 x = +0.342에
있어서, 그보다 앞에 두면 스캔을 가린다. 다만 시뮬레이터 LiDAR는 group 0(월드) 만
레이캐스트하므로 **팔에 의한 가림은 지금 모델에 없다.** 실기에서는 후방 일부가
가려질 수 있다.

팔 자체는 관절 없는 강체 두 개(`payload_deck`, `fr3_arm`)다. 팔을 실제로 움직이는
제어는 이 패키지 밖(젯슨의 FR3 드라이버)이고, 걷기 정책은 팔의 움직임을 관절각이
아니라 도메인 랜덤화로 흡수하도록 학습했다. 자세한 내용은
`rl_training/README.md`의 "B2 + FR3" 절.

## 기동 시퀀스

실기와 같은 순서다.

1. **damping** — kp=0, kd=10. 안전 상태.
2. **stand** — kp=400, kd=8로 1초에 웅크림 자세, 3초에 기립 자세까지 보간.
   Unitree의 B2 `FixStand` 스크립트 그대로다.
3. **policy** — 50 Hz로 정책 실행. kp/kd는 ONNX 메타데이터 값(160/5).

시뮬레이터는 `/lowcmd`가 처음 올 때까지 스폰 자세를 kp=400으로 붙잡는다. 전원만
들어온 83 kg 로봇은 그냥 주저앉는데, 그렇게 무너진 자세에서는 기립 스크립트가
복구하지 못한다. 실기가 컨트롤러 연결 전에 이미 네 발로 서 있는 상태에 대응한다.
`startup_hold_kp` / `startup_hold_kd`로 조정한다.

## 실기 전환

`b2_policy_runner`는 그대로 쓴다. `b2_sim`을 끄고 실제 B2에 DDS를 연결하면 된다
(`b2_driver/README.md`의 CycloneDDS 설정 참고). 다만 실기에 올리기 전에 확인할 것:

- 정책은 **평지(`Unitree-B2-Flat`, 팔 탑재는 `Unitree-B2Arm-Flat`)** 에서만
  학습됐다. 계단·경사는 범위 밖이다.
- 안전을 위해 `max_lin_vel_x` 등 명령 클램프를 먼저 낮춘다.
- 첫 시도는 반드시 로봇을 매단 상태에서 한다.
