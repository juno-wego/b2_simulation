"""Put the b2sim virtualenv on `sys.path` when it is not there already.

`mujoco` and `onnxruntime` are not apt packages and deliberately do not live in
the system python.  `b2_env.sh` exports a PYTHONPATH that reaches them, but a
node launched from a terminal that never sourced it -- a fresh login shell, a
systemd unit, an IDE -- dies on `import mujoco` with nothing but a
ModuleNotFoundError to go on.  That is a bad failure for something a person hits
every time they reboot.

Importing this module first makes the node find its own dependencies:

    import b2_mujoco._venv  # noqa: F401  -- must precede `import mujoco`
    import mujoco

It is a no-op when the environment is already correct, so sourcing `b2_env.sh`
still behaves exactly as before.  The virtualenv is created with
`--system-site-packages` and carries no numpy of its own, so adding it cannot
shadow the numpy, rclpy or message packages that must come from ROS.
"""

from __future__ import annotations

import os
import site
import sys
from importlib.util import find_spec
from pathlib import Path

_VENV_DIRNAME = ".venv-b2sim"
_REQUIRED = ("mujoco", "onnxruntime")


def _candidates() -> list[Path]:
  """Where the virtualenv might be, most explicit first."""
  seen: list[Path] = []

  def add(path: Path | None) -> None:
    if path and path not in seen:
      seen.append(path)

  # b2_env.sh names it, so honour that above all else.
  env = os.environ.get("B2_VENV")
  add(Path(env).expanduser() if env else None)

  # The venv sits at the workspace root, one level above install/ or build/.
  for var in ("COLCON_PREFIX_PATH", "AMENT_PREFIX_PATH"):
    for entry in os.environ.get(var, "").split(os.pathsep):
      if entry:
        add(Path(entry).parent / _VENV_DIRNAME)

  add(Path.home() / "shalom_ws" / _VENV_DIRNAME)
  return seen


def ensure() -> None:
  """Make `_REQUIRED` importable, or raise with something actionable."""
  missing = [m for m in _REQUIRED if find_spec(m) is None]
  if not missing:
    return

  for venv in _candidates():
    packages = venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if not packages.is_dir():
      continue
    site.addsitedir(str(packages))
    missing = [m for m in _REQUIRED if find_spec(m) is None]
    if not missing:
      return

  raise ImportError(
    f"b2_mujoco needs {', '.join(missing)}, which are not importable.\n"
    f"They live in a virtualenv (looked in: "
    f"{', '.join(str(p) for p in _candidates())}).\n"
    "Fix it with either of:\n"
    "  source ~/shalom_ws/src/b2_simulation/mujoco/b2_mujoco/b2_env.sh\n"
    "  python3 -m venv --system-site-packages ~/shalom_ws/.venv-b2sim && \\\n"
    "      ~/shalom_ws/.venv-b2sim/bin/pip install mujoco onnxruntime"
  )


ensure()
