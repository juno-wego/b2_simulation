#!/usr/bin/env python3
"""Generate the mjlab-flavoured B2 MJCF from Unitree's official model.

Training physics must match the deployment simulator, so the geometry, masses and
joint limits are taken from `unitree_mujoco/unitree_robots/b2/b2.xml` untouched.
Only what mjlab needs to resolve its configs is added:

  * every collision geom named `<body><n>_collision`, so CollisionCfg and
    ContactSensorCfg regexes match (`.*_collision`, `^[FR][LR]_foot_collision$`,
    `.*_collision\\d*$`),
  * a foot site per leg named FL/FR/RL/RR, used by foot_height, foot_clearance
    and foot_slip,
  * mjlab's IMU sensor names (imu_ang_vel, imu_lin_vel, imu_lin_acc, root_angmom),
  * no <actuator>/<keyframe>: mjlab injects position actuators from the EntityCfg.

Usage:
    python make_b2_asset.py [--src <unitree_mujoco b2 dir>] [--dst <mjlab xmls dir>]
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# unitree_mujoco is Unitree's official checkout and lives outside this workspace.
DEFAULT_SRC = Path("~/ros2_ws/src/unitree_mujoco/unitree_robots/b2").expanduser()
DEFAULT_DST = Path(
    "~/unitree_rl_mjlab/src/assets/robots/unitree_b2/xmls"
).expanduser()

LEGS = ("FL", "FR", "RL", "RR")


def collision_geoms(body):
    return [g for g in body.findall("geom") if g.get("class") == "collision"]


def short_name(body_name):
    # Unitree's Go2 model names base collisions base1_collision..base3_collision,
    # not base_link1_collision; keep that convention so shared regexes work.
    return "base" if body_name == "base_link" else body_name


def build(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    if (dst / "assets").exists():
        shutil.rmtree(dst / "assets")
    shutil.copytree(src / "assets", dst / "assets")

    tree = ET.parse(src / "b2.xml")
    root = tree.getroot()
    worldbody = root.find("worldbody")

    base = worldbody.find("body")
    assert base.get("name") == "base_link"
    base.set("pos", "0 0 0.58")  # Stand height for the flat-ground default pose.

    named, sites = [], []
    for body in worldbody.iter("body"):
        body_name = body.get("name")
        geoms = collision_geoms(body)
        if not geoms:
            continue

        foot = None
        if body_name.endswith("_calf"):
            leg = body_name[:2]
            assert leg in LEGS, leg
            # The foot is the calf's sphere collider at the far end of the shank.
            spheres = [g for g in geoms if g.get("type") == "sphere"]
            assert len(spheres) == 1, (body_name, len(spheres))
            foot = spheres[0]
            foot.set("name", f"{leg}_foot_collision")
            named.append(foot.get("name"))
            # mjlab reads foot height and velocity from a site, not the geom.
            site = ET.SubElement(body, "site")
            site.set("name", leg)
            site.set("pos", foot.get("pos"))
            site.set("type", "sphere")
            site.set("size", foot.get("size"))
            sites.append((leg, foot.get("pos")))

        rest = [g for g in geoms if g is not foot]
        stem = short_name(body_name)
        for i, g in enumerate(rest, start=1):
            g.set("name", f"{stem}_collision" if len(rest) == 1 else f"{stem}{i}_collision")
            named.append(g.get("name"))

    # Sites must precede child bodies in MJCF; ElementTree appends, so re-sort.
    for body in worldbody.iter("body"):
        kids = list(body)
        body[:] = [k for k in kids if k.tag != "body"] + [k for k in kids if k.tag == "body"]

    for tag in ("actuator", "keyframe", "sensor"):
        for el in root.findall(tag):
            root.remove(el)

    sensor = ET.SubElement(root, "sensor")
    for tag, attrs in (
        ("gyro", {"name": "imu_ang_vel", "site": "imu"}),
        ("velocimeter", {"name": "imu_lin_vel", "site": "imu"}),
        ("accelerometer", {"name": "imu_lin_acc", "site": "imu"}),
        ("subtreeangmom", {"name": "root_angmom", "body": "base_link"}),
    ):
        el = ET.SubElement(sensor, tag)
        for k, v in attrs.items():
            el.set(k, v)

    ET.indent(tree, space="  ")
    out = dst / "b2.xml"
    tree.write(out, encoding="unicode", xml_declaration=False)

    print(f"wrote {out}")
    print(f"named {len(named)} collision geoms, added {len(sites)} foot sites")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="unitree_mujoco B2 model directory")
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST,
                    help="destination xmls directory inside unitree_rl_mjlab")
    args = ap.parse_args()
    if not (args.src / "b2.xml").is_file():
        raise SystemExit(f"no b2.xml under {args.src}")
    build(args.src, args.dst)


if __name__ == "__main__":
    main()
