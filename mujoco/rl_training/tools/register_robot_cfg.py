#!/usr/bin/env python3
"""Add a robot factory export to mjlab's assets/robots/__init__.py (idempotent).

    register_robot_cfg.py <init.py> <package> <module> <factory>
"""

import pathlib
import sys


def main() -> None:
    init, pkg, mod, fn = sys.argv[1:5]
    p = pathlib.Path(init)
    s = p.read_text()
    if fn in s:
        return
    add = f"from .{pkg}.{mod} import (\n  {fn} as {fn},\n)\n\n"
    anchor = "from .unitree_a2.a2_constants import"
    p.write_text(s.replace(anchor, add + anchor, 1) if anchor in s else add + s)


if __name__ == "__main__":
    main()
