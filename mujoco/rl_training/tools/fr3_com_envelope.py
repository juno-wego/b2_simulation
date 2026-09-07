"""Sample the FR3's real centre-of-mass envelope from FAIRINO's own URDF.

The training config currently uses an estimate ("whole-arm COM sits at roughly
0.35 of reach").  This replaces it with the measured link masses and joint
origins.
"""
import itertools
import sys
import xml.etree.ElementTree as ET

import numpy as np


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def T(R, t):
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


root = ET.parse(sys.argv[1]).getroot()

links = {}
for l in root.findall("link"):
    i = l.find("inertial")
    if i is None:
        continue
    o = i.find("origin")
    links[l.get("name")] = (
        float(i.find("mass").get("value")),
        np.array([float(v) for v in o.get("xyz").split()]),
    )

joints = []
for j in root.findall("joint"):
    o = j.find("origin")
    lim = j.find("limit")
    joints.append({
        "name": j.get("name"),
        "parent": j.find("parent").get("link"),
        "child": j.find("child").get("link"),
        "xyz": np.array([float(v) for v in o.get("xyz").split()]),
        "rpy": np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]),
        "axis": np.array([float(v) for v in j.find("axis").get("xyz").split()]),
        "lo": float(lim.get("lower")),
        "hi": float(lim.get("upper")),
    })

TOTAL = sum(m for m, _ in links.values())
print(f"arm total mass: {TOTAL:.3f} kg over {len(links)} links, {len(joints)} joints")
print(f"link masses: " + ", ".join(f"{n.replace('_link','')} {m:.2f}" for n, (m, _) in links.items()))


def axis_rot(axis, q):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def com(q, tool_mass=0.0):
    """Whole-arm COM in the base_link frame; tool_mass rides on the flange."""
    frames = {"base_link": np.eye(4)}
    for j, qi in zip(joints, q):
        frames[j["child"]] = (
            frames[j["parent"]] @ T(rpy(*j["rpy"]), j["xyz"]) @ T(axis_rot(j["axis"], qi), np.zeros(3))
        )
    num, den = np.zeros(3), 0.0
    for name, (m, c) in links.items():
        p = frames[name] @ np.append(c, 1.0)
        num += m * p[:3]
        den += m
    if tool_mass:
        flange = frames[joints[-1]["child"]] @ np.array([0, 0, 0, 1.0])
        num += tool_mass * flange[:3]
        den += tool_mass
    return num / den


# Grid-sample the joints that actually swing the mass around.  j5/j6 move the
# 1.9 kg wrist only, so they are sampled coarsely.
grids = []
for k, j in enumerate(joints):
    n = 9 if k < 4 else 3
    grids.append(np.linspace(j["lo"], j["hi"], n))

for tool in (0.0, 3.0):
    pts = np.array([com(q, tool) for q in itertools.product(*grids)])
    lo, hi = pts.min(0), pts.max(0)
    label = f"tool {tool:.0f} kg"
    print(f"\n{label}:  {len(pts)} poses, effective mass {TOTAL + tool:.2f} kg")
    for i, ax in enumerate("xyz"):
        print(f"  COM {ax}: {lo[i]:+.3f} .. {hi[i]:+.3f} m   (span {hi[i]-lo[i]:.3f})")
    print(f"  horizontal radius max: {np.hypot(pts[:,0], pts[:,1]).max():.3f} m")
