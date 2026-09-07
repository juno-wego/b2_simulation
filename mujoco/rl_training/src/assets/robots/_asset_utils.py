"""Compatibility shim for `update_assets`.

mjlab exposed this helper as `mjlab.utils.os.update_assets` up to 1.2.x and
dropped it in 1.4.0, but every robot's `*_constants.py` in unitree_rl_mjlab still
imports it.  Providing it here keeps those files working against current mjlab
without vendoring a whole mjlab version.
"""

from __future__ import annotations

from pathlib import Path


def update_assets(assets: dict[str, bytes], asset_dir: Path, meshdir: str) -> None:
    """Populate ``assets`` with every file under ``asset_dir``.

    Keys are the paths MuJoCo will look them up by: relative to ``asset_dir``,
    prefixed with the model's ``meshdir`` unless that is empty or ".".
    """
    meshdir_path = Path(meshdir)
    for file_path in asset_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(asset_dir)
        asset_key = rel_path if meshdir in {"", "."} else meshdir_path / rel_path
        assets[asset_key.as_posix()] = file_path.read_bytes()
