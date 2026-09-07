# mujoco

MuJoCo 기반 B2 시뮬레이션. 두 부분으로 나뉜다.

| 디렉터리 | 내용 |
|---|---|
| [`b2_mujoco/`](b2_mujoco/) | ROS 2 패키지 — 시뮬레이터 노드, 정책 러너, LiDAR, 씬, Nav2/SLAM 설정 |
| [`rl_training/`](rl_training/) | B2 velocity policy 학습 자산 (`unitree_rl_mjlab`에 설치해서 사용) |

## 왜 MuJoCo인가

B2용 공개 velocity policy가 없어서 직접 학습해야 했다. 학습 프레임워크로
[`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab)을 골랐고,
이유는 하나다: **mjlab은 MuJoCo로 학습하고 배포 대상도 MuJoCo라 sim-to-sim 격차가
0이다.** 대안이던 `unitree_rl_lab`은 Isaac Sim이 필요하고 (이 머신에 없음), B2
정책 가중치도 저장소에 없다.

## 실행

```bash
source ~/shalom_ws/src/b2_simulation/mujoco/b2_mujoco/b2_env.sh

ros2 launch b2_mujoco b2_slam.launch.py    # SLAM + Nav2 전체
ros2 launch b2_mujoco b2_sim.launch.py     # 걷기만
```

설치·검증 결과·튜닝은 [`b2_mujoco/README.md`](b2_mujoco/README.md)에 있다.

## 정책을 다시 학습하려면

```bash
cd rl_training
./install.sh ~/unitree_rl_mjlab      # B2 에셋과 태스크를 mjlab에 설치

cd ~/unitree_rl_mjlab
WANDB_MODE=offline .venv/bin/python scripts/train.py Unitree-B2-Flat \
  --env.scene.num-envs=4096 --agent.max-iterations=6000
```

RTX 4060 Laptop(8 GB)에서 4096 envs 기준 약 1.17 s/iteration, 6000 iteration에
2시간. 진행 상황은 `ros2 run b2_mujoco b2_train_progress`.
