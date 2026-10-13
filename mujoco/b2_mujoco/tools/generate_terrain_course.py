"""Regenerate the course height field; no MuJoCo or image libraries required.

Run from the package root:
    python3 tools/generate_terrain_course.py

The compact elliptical hill has zero slope and curvature at its boundary.
MuJoCo uses one height-field surface for rendering, contacts and LiDAR rays.
Peak height is configured separately from the normalized elevation samples.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


HILL_CENTER = (-3.0, 0.0)
HILL_RADII = (7.0, 4.4)
# 0.2 m keeps 32x720 LiDAR raycasts within the 10 Hz scan budget while the
# smooth hill's height approximation remains within about 1.3 mm.
GRID_SPACING = 0.2
WORLD_FILE = Path(__file__).resolve().parents[1] / "models" / "terrain_course.xml"


def generate() -> Path:
    world = ET.parse(WORLD_FILE).getroot()
    hfield = world.find("asset/hfield[@name='course_height']")
    if hfield is None:
        raise ValueError("terrain_course.xml has no course_height asset")
    radius_x, radius_y, height, _ = map(float, hfield.attrib["size"].split())
    ncol = round(2 * radius_x / GRID_SPACING) + 1
    nrow = round(2 * radius_y / GRID_SPACING) + 1
    values = []
    for row in range(nrow):
        y = -radius_y + 2 * radius_y * row / (nrow - 1)
        for col in range(ncol):
            x = -radius_x + 2 * radius_x * col / (ncol - 1)
            r2 = ((x - HILL_CENTER[0]) / HILL_RADII[0]) ** 2
            r2 += ((y - HILL_CENTER[1]) / HILL_RADII[1]) ** 2
            values.append(max(0.0, 1.0 - r2) ** 3)

    # Inline data keeps the existing scene self-contained, including when it is
    # loaded through b2.xml's meshdir or from a symlink install.
    rows = [" ".join(format(value, ".8g") for value in values[start:start + ncol])
            for start in range(0, len(values), ncol)]
    elevation = "\n             ".join(rows)
    element = (f'<hfield name="course_height" size="{hfield.attrib["size"]}" '
               f'nrow="{nrow}" ncol="{ncol}"\n'
               f'            elevation="{elevation}"/>')
    xml = WORLD_FILE.read_text()
    xml, count = re.subn(r'<hfield\b(?=[^>]*name="course_height")[^>]*/>',
                        lambda _: element, xml)
    if count != 1:
        raise ValueError("expected one self-closing course_height element")
    WORLD_FILE.write_text(xml)
    peak_gradient = 96 / (25 * math.sqrt(5))
    slopes = [math.degrees(math.atan(peak_gradient * height / radius))
              for radius in HILL_RADII]
    print(f"Generated {WORLD_FILE}: {nrow}x{ncol}, height {height:g} m, "
          f"maximum slopes X={slopes[0]:.1f}, Y={slopes[1]:.1f} degrees")
    return WORLD_FILE


if __name__ == "__main__":
    generate()
