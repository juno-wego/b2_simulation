# rl_training

`unitree_rl_mjlab`에 B2를 추가하는 파일들. mjlab은 Go2, A2, As2, G1, R1, H1_2, H2를
지원하지만 B2는 없다.

업스트림 클론(`~/unitree_rl_mjlab`)에만 두면 다시 클론할 때 사라지므로 여기에
보관한다.

**로봇이 둘이다.** 맨몸 B2와, 등에 FAIRINO FR3 협동로봇 팔을 얹은 B2. 두 계보는
설정 객체를 하나도 공유하지 않고 로그 디렉터리도 다르다(`b2_velocity` /
`b2_arm_velocity`). 한쪽을 손봐도 다른 쪽이 절대 망가지지 않는다.

```
src/assets/robots/unitree_b2/b2_constants.py         맨몸 B2 에셋
src/assets/robots/unitree_b2_arm/b2_arm_constants.py B2 + FR3 에셋
src/tasks/velocity/config/b2/                        Unitree-B2-Flat / -Rough
src/tasks/velocity/config/b2_arm/                    Unitree-B2Arm-Flat / -Rough
tools/make_b2_asset.py                               공식 MJCF -> mjlab 규약 변환
tools/make_b2_arm_asset.py                           위에 FR3 페이로드 부착
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

## B2 + FR3 (팔 탑재)

FR3는 자중 15 kg, 가반하중 3 kg, 리치 622 mm, 컨트롤 박스 2.5 kg다. 마운트 플레이트와
DC-DC(B2 배터리 → FR3의 48 V DC 입력)까지 더하면 **21.5 kg**이 등 위에 올라간다.
B2의 이동 정격 40 kg 안이라 무게는 문제가 아니다. **무게중심이 문제다.**

| | 맨몸 B2 | B2 + FR3 |
|---|---|---|
| 총질량 | 83.5 kg | 105.0 kg |
| 전신 COM 높이 (기립 시) | 0.553 m | 0.612 m |

맨몸 정책의 `base_com` 랜덤화는 ±0.05 m뿐이다. 15 kg짜리 팔이 0.33 m 위에서
622 mm를 휘두르면 COM이 그 범위를 한참 벗어난다. 그래서 정책을 따로 학습한다.

### 팔을 어떻게 모델링했나

FR3의 URDF/MJCF가 없다. 관절 체인 대신 **강체 두 개**로 근사했다.

```
payload_deck   컨트롤 박스 + 마운트 플레이트 + DC-DC   6.5 kg   (트렁크 상판 z=0.10)
fr3_arm        팔 본체                                15.0 kg  (접힌 상태 COM, z=0.33)
```

팔이 *움직인다*는 사실은 도메인 랜덤화로 들어간다
(`src/tasks/velocity/config/b2_arm/payload.py`).

| 항목 | 범위 | 근거 |
|---|---|---|
| `arm_com` — 팔 COM 위치 | x, y ±0.30 m / z −0.25~+0.12 m | 6축 협동로봇의 전체 COM은 리치의 약 0.35 지점. 0.35×0.622 = 0.22 m, 여기에 3 kg 툴이 0.622 m 끝에 붙으면 합성 COM은 0.285 m |
| `arm_mass` | 14.3 ~ 19.5 kg | 맨팔 ~ 팔 + 3 kg 툴 + 무거운 그리퍼 |
| `deck_mass` | 5.5 ~ 8.1 kg | 컨트롤 박스 2.5 kg은 스펙, 플레이트·DC-DC는 추정치 |
| `arm_reaction` | 힘 ±30 N, 토크 ±25 N·m, 0.3~1.5 s | 3 kg 페이로드를 1 m/s에서 0.1 s만에 급정지하면 30 N. J1 슬루(≈0.94 kg·m²)를 5 rad/s²로 돌리면 5 N·m, 0.6 m 지점의 툴을 제동하면 18 N·m |

앞의 셋은 `startup`, 마지막 하나만 `step`이다. 팔은 **로봇이 걷는 동안에도** 움직이기
때문이다.

x, y를 원기둥이 아니라 정육면체로 샘플링하므로 대각선 방향(0.42 m)은 실제 리치를
넘는다. 의도적으로 보수적인 쪽이다.

### 맨몸 설정과 다른 점

보상 *구조*는 건드리지 않았다. 실제로 걷는 정책을 만들어 낸 건 그 보상 조합이고,
21.5 kg을 얹는 것과 동시에 보상 모양까지 바꾸면 결과가 나빴을 때 원인을 못 가린다.

바꾼 것은 셋뿐이다.

| | 맨몸 | 팔 탑재 | 이유 |
|---|---|---|---|
| `body_orientation_l2` | −1.0 | **−2.0** | 트렁크가 1° 기울면 팔 COM이 0.33 m × sin 1° = 6 mm 움직이고, LiDAR도 같은 각도로 흔들린다 |
| `body_ang_vel` | −0.05 | **−0.1** | 위와 같다. 트렁크 각속도가 팔에서 증폭된다 |
| 커리큘럼 상한 | 2.0 / 1.0 / 1.5 | **1.5 / 0.8 / 1.2** | 105 kg에 높은 COM. 최고 속도보다 여유가 낫다. Nav2가 요구하는 건 최대 0.9 m/s다 |

kp/kd는 **160/5 그대로**다. 21.5 kg 때문에 다리가 처지는 건 게인을 올리면 쉽게 잡히지만,
그 게인은 Unitree가 B2 실기에서 검증한 값이고 배포 쪽은 ONNX 메타데이터에서 그대로
읽어 간다. 처짐은 정책이 관절 목표를 다르게 내서 보상한다. 실기에서 공짜다.

기준 자세도 `(0, 0.8, -1.5)` 그대로다. `FixStand` → 정책 인계가 똑같이 매끄러워야 한다.
스폰 높이만 0.545 → 0.555 m로 올렸다. 안 그러면 처진 발이 바닥을 파고든 채 시작한다.

### 학습

```bash
cd ~/unitree_rl_mjlab
WANDB_MODE=offline .venv/bin/python scripts/train.py Unitree-B2Arm-Flat \
    --env.scene.num-envs=4096 --agent.max-iterations=15000 --agent.logger tensorboard
```

RTX 4060 Laptop(8 GB), 4096 envs에서 약 1.35 s/iteration — 15000 iteration에 약 5.5시간.

```bash
cp ~/unitree_rl_mjlab/logs/rsl_rl/b2_arm_velocity/<run>/policy.onnx \
   ~/shalom_ws/src/b2_simulation/mujoco/b2_mujoco/policy/b2_arm_velocity.onnx
```

**맨몸 정책 파일(`b2_velocity.onnx`)에 덮어쓰지 말 것.** 다른 로봇이다.

### 배포 모델도 같은 스크립트가 만든다

`make_b2_arm_asset.py`는 학습 에셋과 배포 에셋을 **둘 다** 만든다.

```bash
# 학습용 (mjlab)
python3 tools/make_b2_arm_asset.py

# 배포용 (b2_mujoco) — Unitree 공식 모델에 같은 페이로드를 주입
python3 tools/make_b2_arm_asset.py \
    --deploy-in  ../b2_mujoco/models/b2.xml \
    --deploy-out ../b2_mujoco/models/b2_arm.xml
```

페이로드 숫자가 한 군데(스크립트 상단)에만 있어야 한다. 학습이 팔을 15 kg으로 알고
배포 시뮬레이터가 20 kg으로 알면, 학습한 적 없는 로봇으로 정책을 검증하는 셈이 된다.

## 학습 참고

`velocity_env_cfg.py`의 커리큘럼이 **iteration 5000에서 명령 범위를 넓힌다**
(`lin_vel_x` 최대 1.0 → 2.0 m/s). 그 지점에서 `error_vel_xy`가 0.43에서 0.79로
뛰고 reward가 47에서 35로 떨어지는데, 성능 저하가 아니라 채점 난이도가 올라간
것이다. 로그를 볼 때 헷갈리기 쉬우니 주의.
