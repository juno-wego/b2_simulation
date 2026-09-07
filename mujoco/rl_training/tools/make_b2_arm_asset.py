#!/usr/bin/env python3
"""Bolt a FAIRINO FR3 cobot onto a B2 MJCF.

One script, one set of payload numbers, two outputs:

  --dst        the *training* asset.  Runs `make_b2_asset.build()` first so the
               quadruped half is byte-identical to the plain B2 mjlab asset, then
               injects the payload.  ->  b2_arm.xml in unitree_rl_mjlab.
  --deploy-in / --deploy-out
               the *deployment* asset.  Injects the arm into
               b2_mujoco/models/b2.xml (Unitree's official model, with its
               actuators, sensors and SDK motor order intact).  Here the arm is
               **articulated**: six hinge joints with the URDF's real masses and
               inertia tensors, and a position actuator on each, so the
               simulator moves the arm the way the robot does and reports joint
               state the way the robot does.

Both must be generated from here.  If training thinks the arm weighs 15 kg and
the deployment simulator thinks it weighs 20, the policy is being validated
against a robot it was not trained on -- the exact failure mode this whole
separate lineage exists to avoid.

## Why training and deployment differ here

Training gets a rigid proxy; the deployment simulator gets the real chain. That
is deliberate, and it is the one place these two models are allowed to disagree.

Training runs 4096 robots at once and needs the arm only as a disturbance: what
the walking policy has to reject is the mass, where the centre of mass is, and
the wrench a moving arm dumps into the trunk. A rigid body whose position is
randomized over the arm's whole reach delivers all three, costs six fewer joints
per robot, and covers poses in one episode that an articulated arm would take a
whole trajectory to visit.

The deployment simulator stands in for the robot, so it has to *be* the robot:
the arm moves when commanded and reports where it is, because the control
station and MoveIt2 both need that and neither should be able to tell the
simulator from hardware. The policy is unaffected - every pose the articulated
arm can reach is inside the envelope it trained against.

## What the payload is (training)

For training the arm is modelled as **two rigid bodies** rather than an
articulated chain:

  payload_deck   the parts that never move relative to the trunk --
                 control box (2.5 kg, 245 x 180 x 44.5 mm per FAIRINO's spec),
                 mounting plate and the 58V -> 48V DC-DC converter.
  fr3_arm        the arm itself (15 kg), placed at the centre of mass it has
                 when stowed.

The arm's *pose dependence* is then covered in the task config by randomizing
`fr3_arm`'s mount position over the envelope its centre of mass actually sweeps
(see `tasks/velocity/config/b2_arm/payload.py`).  A rigid proxy with a swept COM
is the right level of fidelity here: what the walking policy has to reject is the
mass, the COM excursion and the reaction wrench, none of which need the arm's
joint angles.

## Where it sits

base_link's trunk collision box is a 0.50 x 0.28 x 0.15 m box centred on the body
origin, so the deck sits just above it at z = 0.10.  x = -0.05 keeps the arm
behind the front LiDAR mast (a 0.076 m radius cylinder at x = 0.342, z = 0.159)
so a stowed arm does not occlude the scan the SLAM stack depends on.

Usage:
    # training asset
    python make_b2_arm_asset.py [--src <unitree_mujoco b2 dir>] [--dst <xmls dir>]

    # deployment asset
    python make_b2_arm_asset.py --deploy-in  .../b2_mujoco/models/b2.xml \
                                --deploy-out .../b2_mujoco/models/b2_arm.xml
"""

import argparse
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_SRC = Path("~/ros2_ws/src/unitree_mujoco/unitree_robots/b2").expanduser()
DEFAULT_DST = Path(
    "~/unitree_rl_mjlab/src/assets/robots/unitree_b2_arm/xmls"
).expanduser()

# Mount frame on the trunk top plate, in base_link coordinates.
MOUNT = (-0.05, 0.0, 0.10)

# Control box 2.5 kg (FAIRINO spec) + mounting plate ~2.5 kg + DC-DC and loom
# ~1.5 kg.  Box inertia for a 0.30 x 0.22 x 0.06 m slab.
DECK_MASS = 6.5
DECK_HALF = (0.15, 0.11, 0.03)

# FR3 arm, 15 kg (FAIRINO spec), 622 mm reach.  Stowed, the six links fold into
# roughly a 0.25 x 0.25 x 0.45 m volume whose centre sits 0.20 m above the deck;
# J1/J2 hold most of the mass so the COM stays low in that volume.
ARM_MASS = 15.0
ARM_HALF = (0.125, 0.125, 0.22)
ARM_COM_ABOVE_DECK = 0.20


def box_inertia(mass, half):
    hx, hy, hz = half
    k = mass / 3.0  # m/12 * (2h)^2 == m/3 * h^2
    return (k * (hy * hy + hz * hz), k * (hx * hx + hz * hz), k * (hx * hx + hy * hy))


def fmt(vals):
    return " ".join(f"{v:g}" for v in vals)


def _add_body(parent, name, pos, mass, half, rgba):
    body = ET.SubElement(parent, "body")
    body.set("name", name)
    body.set("pos", fmt(pos))
    inertial = ET.SubElement(body, "inertial")
    inertial.set("pos", "0 0 0")
    inertial.set("mass", f"{mass:g}")
    inertial.set("diaginertia", fmt(box_inertia(mass, half)))
    for cls, extra in (
        ("visual", {"rgba": rgba}),
        ("collision", {"name": f"{name}_collision"}),
    ):
        g = ET.SubElement(body, "geom")
        g.set("type", "box")
        g.set("size", fmt(half))
        g.set("class", cls)
        for k, v in extra.items():
            g.set(k, v)
    return body


# FAIRINO FR3, straight out of fairino3_v6.urdf: joint origin, then a hinge
# about the child frame's z, then the link's real mass and inertia tensor.
# `radius`/`half` only size the collision capsule - the mass properties below
# are what the physics uses, and they are the vendor's.
FR3_LINKS = [
    # name,          xyz,                 rpy,           lo,      hi,    mass,  diaginertia,               com
    ("shoulder_link", (0.0, 0.0, 0.0),      (0, 0, 0),   -3.0543, 3.0543, 2.240, (0.00304, 0.00224, 0.00296), (-0.00005, -0.01592, 0.00226)),
    ("upperarm_link", (0.0, 0.0, 0.14),     (1.5708, 0, 0), -4.6251, 1.4835, 4.940, (0.00681, 0.09321, 0.09151), (0.13939, 0.0, 0.09954)),
    ("forearm_link",  (-0.28, 0.0, 0.0),    (0, 0, 0),   -2.8274, 2.8274, 2.290, (0.00264, 0.02075, 0.02039), (0.05899, 0.00008, 0.01299)),
    ("wrist1_link",   (-0.24001, 0.0, 0.0), (0, 0, 0),   -4.6251, 1.4835, 1.560, (0.00163, 0.00158, 0.00126), (0.00005, -0.00233, 0.01467)),
    ("wrist2_link",   (0.0, 0.0, 0.102),    (1.5708, 0, 0), -3.0543, 3.0543, 1.560, (0.00163, 0.00158, 0.00126), (-0.00005, 0.00233, 0.01467)),
    ("wrist3_link",   (0.0, 0.0, 0.102),    (-1.5708, 0, 0), -3.0543, 3.0543, 0.360, (0.00023, 0.00023, 0.00036), (-0.00055, -0.00111, -0.02005)),
]

# The arm's own base, fixed to the deck. 0.784 kg from the URDF.
FR3_BASE_MASS = 0.784
FR3_BASE_INERTIA = (0.00139, 0.00138, 0.00198)

# Gains for the position actuators. Not the FR3's real servo gains - those are
# inside FAIRINO's controller and are not published - but stiff enough that the
# arm holds a commanded pose against gravity and settles in well under a
# second, which is what the simulator has to reproduce for the station.
FR3_KP = 900.0
FR3_KD = 60.0
FR3_EFFORT = 150.0


def inject_articulated_arm(root: ET.Element) -> None:
    """Add the real six-joint FR3 to base_link, on the payload deck."""
    base = root.find("worldbody").find("body")
    assert base.get("name") == "base_link", base.get("name")

    # FAIRINO's own STLs, copied into the model's assets directory. They are in
    # metres already - the URDF declares no scale - so nothing is rescaled here.
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for stem in ("base_link", *[l[0] for l in FR3_LINKS]):
        mesh = ET.SubElement(asset, "mesh")
        mesh.set("name", f"fr3_{stem}")
        mesh.set("file", f"fr3_{stem}.STL")

    mx, my, mz = MOUNT
    arm = ET.SubElement(base, "body")
    arm.set("name", "fr3_base_link")
    # Sits on top of the deck, not inside it.
    arm.set("pos", fmt((mx, my, mz + DECK_HALF[2])))
    inertial = ET.SubElement(arm, "inertial")
    inertial.set("pos", "0 0 0.033")
    inertial.set("mass", f"{FR3_BASE_MASS:g}")
    inertial.set("diaginertia", fmt(FR3_BASE_INERTIA))
    gv = ET.SubElement(arm, "geom")
    gv.set("type", "mesh")
    gv.set("mesh", "fr3_base_link")
    gv.set("group", "1")
    gv.set("contype", "0")
    gv.set("conaffinity", "0")
    gv.set("density", "0")
    gv.set("rgba", "0.85 0.85 0.88 1")

    gc = ET.SubElement(arm, "geom")
    gc.set("name", "fr3_base_collision")
    gc.set("type", "cylinder")
    gc.set("size", "0.064 0.04")
    gc.set("pos", "0 0 0.04")
    gc.set("class", "collision")

    parent = arm
    for name, xyz, rpy, lo, hi, mass, inertia, com in FR3_LINKS:
        body = ET.SubElement(parent, "body")
        body.set("name", f"fr3_{name}")
        body.set("pos", fmt(xyz))
        if any(rpy):
            body.set("euler", fmt(rpy))

        bi = ET.SubElement(body, "inertial")
        bi.set("pos", fmt(com))
        bi.set("mass", f"{mass:g}")
        bi.set("diaginertia", fmt(inertia))

        j = ET.SubElement(body, "joint")
        j.set("name", f"fr3_{name.split('_')[0]}")
        j.set("type", "hinge")
        j.set("axis", "0 0 1")
        j.set("range", f"{lo:g} {hi:g}")
        j.set("damping", "2")
        j.set("armature", "0.05")

        # Two geoms per link, the same split B2's own model uses.
        #
        # Visual is FAIRINO's own mesh, in group 1 - the group B2's meshes are
        # in and the one the viewer shows by default. `class="visual"` would put
        # it in group 2, where the arm was simply invisible next to a robot that
        # was not.
        #
        # Collision stays a capsule spanning the link rather than the mesh.
        # MuJoCo collides meshes by their convex hull, so a concave arm link
        # gains volume it does not have; a capsule is honest about being an
        # approximation and is far cheaper. What matters is that a collision
        # exists at all - without one the arm carries mass and swings the centre
        # of mass around while passing straight through B2's own body, and
        # nothing reports the contact.
        gv = ET.SubElement(body, "geom")
        gv.set("type", "mesh")
        gv.set("mesh", f"fr3_{name}")
        gv.set("group", "1")
        gv.set("contype", "0")
        gv.set("conaffinity", "0")
        gv.set("density", "0")
        gv.set("rgba", "0.18 0.44 0.84 1")

        gc = ET.SubElement(body, "geom")
        gc.set("name", f"fr3_{name.split('_')[0]}_collision")
        gc.set("type", "capsule")
        gc.set("fromto", "0 0 0 " + fmt(_next_offset(name)))
        gc.set("size", "0.045")
        gc.set("class", "collision")
        parent = body

    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    for name, *_rest in FR3_LINKS:
        jn = f"fr3_{name.split('_')[0]}"
        a = ET.SubElement(act, "position")
        a.set("name", jn)
        a.set("joint", jn)
        a.set("kp", f"{FR3_KP:g}")
        a.set("kv", f"{FR3_KD:g}")
        a.set("forcerange", f"-{FR3_EFFORT:g} {FR3_EFFORT:g}")


def _next_offset(name: str):
    """Where the next joint sits, so a link's capsule spans the real link."""
    order = [l[0] for l in FR3_LINKS]
    i = order.index(name)
    if i + 1 < len(FR3_LINKS):
        return FR3_LINKS[i + 1][1]
    return (0.0, 0.0, 0.05)   # 마지막 링크: 플랜지까지


def inject_payload(root: ET.Element) -> None:
    """Add payload_deck and fr3_arm to a B2 MJCF's base_link, in place."""
    root.set("model", "b2_arm")
    base = root.find("worldbody").find("body")
    assert base.get("name") == "base_link", base.get("name")

    mx, my, mz = MOUNT
    _add_body(base, "payload_deck", MOUNT, DECK_MASS, DECK_HALF, "0.15 0.15 0.17 1")
    _add_body(
        base,
        "fr3_arm",
        (mx, my, mz + DECK_HALF[2] + ARM_COM_ABOVE_DECK),
        ARM_MASS,
        ARM_HALF,
        "0.88 0.88 0.9 1",
    )


def _write(tree: ET.ElementTree, out: Path) -> None:
    ET.indent(tree, space="  ")
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="unicode", xml_declaration=False)
    print(f"wrote {out}")
    print(
        f"  payload {DECK_MASS + ARM_MASS:g} kg "
        f"(deck {DECK_MASS:g} + arm {ARM_MASS:g}) at {fmt(MOUNT)}"
    )


def build_training(src: Path, dst: Path) -> None:
    # Reuse the plain-B2 generator verbatim, writing into our own xmls dir.
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "make_b2_asset", here / "make_b2_asset.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build(src, dst)

    plain = dst / "b2.xml"
    tree = ET.parse(plain)
    inject_payload(tree.getroot())
    _write(tree, dst / "b2_arm.xml")
    plain.unlink()  # only b2_arm.xml belongs in this asset directory


def build_deployment(src_xml: Path, out_xml: Path) -> None:
    """B2 with the deck and a *moving* FR3 on it."""
    tree = ET.parse(src_xml)
    root = tree.getroot()
    root.set("model", "b2_arm")

    base = root.find("worldbody").find("body")
    _add_body(base, "payload_deck", MOUNT, DECK_MASS, DECK_HALF, "0.15 0.15 0.17 1")
    inject_articulated_arm(root)

    ET.indent(tree, space="  ")
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="unicode", xml_declaration=False)
    print(f"wrote {out_xml}")
    print(f"  deck {DECK_MASS:g} kg + articulated FR3 "
          f"{FR3_BASE_MASS + sum(l[5] for l in FR3_LINKS):.3f} kg at {fmt(MOUNT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--deploy-in", type=Path, help="b2_mujoco/models/b2.xml")
    ap.add_argument("--deploy-out", type=Path, help="b2_mujoco/models/b2_arm.xml")
    args = ap.parse_args()

    if args.deploy_in or args.deploy_out:
        if not (args.deploy_in and args.deploy_out):
            raise SystemExit("--deploy-in and --deploy-out must be given together")
        if not args.deploy_in.is_file():
            raise SystemExit(f"no such file: {args.deploy_in}")
        build_deployment(args.deploy_in, args.deploy_out)
        return

    if not (args.src / "b2.xml").is_file():
        raise SystemExit(f"no b2.xml under {args.src}")
    build_training(args.src, args.dst)


if __name__ == "__main__":
    main()
