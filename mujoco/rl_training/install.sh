#!/usr/bin/env bash
# Install the B2 additions into a unitree_rl_mjlab checkout.
#
#   ./install.sh [path-to-unitree_rl_mjlab]
#
# unitree_rl_mjlab ships Go2, A2, As2, G1, R1, H1_2 and H2 -- no B2.  These files
# add it: the robot asset (actuator limits and nominal pose from Unitree's own B2
# deploy config) and the Unitree-B2-Flat / Unitree-B2-Rough velocity tasks.  They
# live here rather than only in the upstream clone so re-cloning does not lose them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MJLAB="${1:-$HOME/unitree_rl_mjlab}"

[ -d "$MJLAB/src/assets/robots" ] || {
  echo "not a unitree_rl_mjlab checkout: $MJLAB" >&2
  echo "usage: $0 [path-to-unitree_rl_mjlab]" >&2
  exit 1
}

echo "installing B2 into $MJLAB"

mkdir -p "$MJLAB/src/assets/robots/unitree_b2" "$MJLAB/src/tasks/velocity/config/b2"
cp "$HERE"/src/assets/robots/unitree_b2/*.py "$MJLAB/src/assets/robots/unitree_b2/"
cp "$HERE"/src/assets/robots/_asset_utils.py "$MJLAB/src/assets/robots/"

# mjlab dropped mjlab.utils.os.update_assets in 1.4.0, but every robot's
# constants file still imports it.  Point them at the shim instead.
# `|| true`: once every file is patched grep exits 1, and `set -o pipefail` would
# take that as a failure and abort a re-run.
grep -rl "from mjlab.utils.os import update_assets" "$MJLAB/src/assets/robots/" 2>/dev/null |
  { while read -r f; do
    sed -i 's|from mjlab.utils.os import update_assets|from src.assets.robots._asset_utils import update_assets|' "$f"
    echo "  patched $(basename "$(dirname "$f")")/$(basename "$f")"
  done; } || true
cp "$HERE"/src/tasks/velocity/config/b2/*.py "$MJLAB/src/tasks/velocity/config/b2/"

# Export the robot factories alongside the ones that ship with mjlab.
INIT="$MJLAB/src/assets/robots/__init__.py"
register_cfg() {  # $1 = package dir, $2 = constants module, $3 = factory name
  if grep -q "$3" "$INIT"; then
    echo "  $3 already registered"
    return
  fi
  python3 "$HERE/tools/register_robot_cfg.py" "$INIT" "$1" "$2" "$3"
  echo "  registered $3 in assets/robots/__init__.py"
}
register_cfg unitree_b2     b2_constants     get_b2_robot_cfg

# The MJCF is generated from Unitree's official model rather than vendored, so the
# training physics cannot drift from the deployment simulator's.
XMLS="$MJLAB/src/assets/robots/unitree_b2/xmls"
if [ ! -f "$XMLS/b2.xml" ]; then
  echo "  generating MJCF from unitree_mujoco..."
  python3 "$HERE/tools/make_b2_asset.py" --dst "$XMLS"
else
  echo "  MJCF already present ($XMLS/b2.xml)"
fi

cat <<'MSG'

done.  Verify and train with:

  cd <unitree_rl_mjlab>
  .venv/bin/python scripts/list_envs.py | grep Unitree-B2
  # bare robot
  WANDB_MODE=offline .venv/bin/python scripts/train.py Unitree-B2-Flat \
      --env.scene.num-envs=4096 --agent.max-iterations=6000

then copy the run's policy.onnx to
b2_simulation/mujoco/b2_mujoco/policy/b2_velocity.onnx.
MSG
