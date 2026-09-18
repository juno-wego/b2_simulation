# rl_training

`unitree_rl_mjlab`에 B2를 추가하는 파일들. mjlab은 Go2, A2, As2, G1, R1, H1_2, H2를
지원하지만 B2는 없다.

업스트림 클론(`~/unitree_rl_mjlab`)에만 두면 다시 클론할 때 사라지므로 여기에
보관한다.

```
src/assets/robots/unitree_b2/b2_constants.py         맨몸 B2 에셋
src/tasks/velocity/config/b2/                        Unitree-B2-Flat / -Rough
tools/make_b2_asset.py                               공식 MJCF -> mjlab 규약 변환
install.sh                                           위 파일들을 mjlab에 설치
```

## 설치

```bash
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git ~/unitree_rl_mjlab
cd ~/unitree_rl_mjlab && python3 -m venv .venv && .venv/bin/pip install -e .

# 저장소 setup.py 의 핀은 현재 mjlab 과 맞지 않는다. 아래로 덮어쓴다.
.venv/bin/pip install "mujoco==3.8.1" "mujoco-warp==3.8.1" "mjlab==1.4.0" "warp-lang==1.14.0"

cd <이 디렉터리> && ./install.sh ~/unitree_rl_mjlab
```

### 버전 조합이 까다롭다

`pip install -e .` 만 하면 최신 패키지가 딸려와 네 군데서 깨진다. 실제로 겪은 것:

| 증상 | 원인 | 해결 |
|---|---|---|
| `mjtEnableBit has no attribute mjENBL_MULTICCD` | mujoco 3.12 가 mujoco-warp 3.5 와 불일치 | `mujoco==3.8.1`, `mujoco-warp==3.8.1` |
| `WarpCodegenKeyError: undefined symbol: xmat` | warp-lang 1.17 이 mujoco-warp 3.8 과 불일치 | `warp-lang==1.14.0` |
| `ImportError: cannot import name 'update_assets'` | mjlab 1.4 가 `mjlab.utils.os.update_assets` 를 제거 | `install.sh` 가 shim 배포 후 import 를 패치 |
| `ValidationError: start_method Extra inputs are not permitted` | wandb 0.29 가 해당 인자를 제거 | 학습 시 `--agent.logger tensorboard` |

`setup.py` 의 `mjlab==1.2.0`, `mujoco-warp==3.5.0` 핀 때문에 pip 이 의존성 경고를 내지만,
동작하는 조합은 위 표 쪽이다. 경고는 무시해도 된다.

`install.sh`가 파일 복사, `assets/robots/__init__.py`에 `get_b2_robot_cfg` 등록,
그리고 MJCF 생성까지 한다. 여러 번 실행해도 안전하다.

## 설계 결정

**MJCF는 벤더링하지 않고 생성한다.** `tools/make_b2_asset.py`가 Unitree 공식
`unitree_mujoco/unitree_robots/b2/b2.xml`에서 만들어 낸다. 지오메트리·질량·관절
한계를 그대로 두고, mjlab이 설정을 해석하는 데 필요한 것만 덧붙인다.

- 이름 붙은 `*_collision` geom (CollisionCfg / ContactSensorCfg 정규식이 매칭되도록)
- 발 site FL/FR/RL/RR (foot_height, foot_clearance, foot_slip이 읽음)
- mjlab IMU 센서 이름 (`imu_ang_vel`, `imu_lin_vel`, `imu_lin_acc`, `root_angmom`)
- `<actuator>`/`<keyframe>` 제거 — mjlab이 EntityCfg에서 position actuator를 주입

이렇게 하면 **학습 물리와 배포 시뮬레이터 물리가 어긋날 수 없다.**

**액추에이터 한계와 기준 자세는 Unitree 자신의 B2 배포 설정에서 가져왔다**
(`unitree_rl_lab`의 `assets/robots/unitree.py`, `deploy/robots/b2/config/config.yaml`).

- hip/thigh 200 Nm, calf 320 Nm (M107 구동계)
- kp 160, kd 5
- 기준 자세 `(hip, thigh, calf) = (0, 0.8, -1.5)` — B2의 `FixStand` FSM이 정책에
  제어권을 넘기기 직전에 유지하는 자세다. 덕분에 정책의 첫 action(=0)이 무동작이
  되어 인계가 매끄럽다.

**Go2 대비 스케일 조정.** B2 다리가 약 1.65배 길어서 보행 주기 0.6 → 0.7 s,
발 들어올림 높이 0.10 → 0.15 m만 바꿨다. 나머지 보상·관측 구조는 검증된 Go2
설정을 그대로 쓴다.

## 학습 참고

`velocity_env_cfg.py`의 커리큘럼이 **iteration 5000에서 명령 범위를 넓힌다**
(`lin_vel_x` 최대 1.0 → 2.0 m/s). 그 지점에서 `error_vel_xy`가 0.43에서 0.79로
뛰고 reward가 47에서 35로 떨어지는데, 성능 저하가 아니라 채점 난이도가 올라간
것이다. 로그를 볼 때 헷갈리기 쉬우니 주의.
