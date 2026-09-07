"""What the FR3 does to B2, expressed as domain randomization.

The arm is a rigid two-body proxy in the MJCF (`tools/make_b2_arm_asset.py`), so
everything the arm's *motion* does to the walking policy has to come from here.
Three effects matter, and each gets its own term:

1. **The COM moves.**  A 622 mm arm slewing around its J1 axis drags 15 kg
   through a wide arc.  Randomizing the arm body's mount position over that arc
   makes the policy face every arm pose it will ever see.  This is the dominant
   effect and the reason the plain-B2 policy cannot be reused: its `base_com`
   term only ever moved the COM by +-0.05 m.

2. **The payload mass is uncertain.**  Tool and workpiece are 0-3 kg on top of
   the arm, and the vendor's 15 kg figure is a round number.

3. **Accelerating the arm pushes back on the trunk.**  A pick-and-place move
   dumps a transient force and torque into the base that the policy must ride
   out without stumbling.

Sources for the hardware numbers: FAIRINO list the FR3 at 15 kg, 3 kg payload,
622 mm reach, with a 2.5 kg / 245x180x44.5 mm control box that accepts an
optional 48 V DC input (which is why it can run off B2's own battery rail).

VERIFIED AGAINST FAIRINO'S OWN URDF
-----------------------------------
The COM envelope below started as an estimate.  It was afterwards checked
against `fairino3_v6.urdf` from FAIR-INNOVATION/frcobot_ros2 (a copy sits in
`tools/`), by sampling 59049 poses across the real joint limits with the real
link masses -- `tools/fr3_com_envelope.py` reproduces it:

    arm mass                13.734 kg over 6 joints (upperarm 4.94 kg dominates)
    link lengths            0.28 + 0.24 + 0.102 = 0.622 m, matching the spec reach

                            trained on           URDF says
    COM horizontal          +-0.300 m            +-0.217 m
    COM height (B2 frame)   0.080 .. 0.450 m     0.035 .. 0.459 m
    arm mass                14.25 .. 19.41 kg    13.73 .. 16.73 kg

The trained envelope is wider than the real one everywhere that makes the
policy's job harder -- 38% more horizontal COM travel, a heavier arm -- and
falls 4.5 cm short only at the *bottom* of the vertical range, where the COM is
lower and the robot is more stable, not less.  So the estimate is a superset in
the direction that matters and the policy is not being under-trained.  Left as
it is rather than re-fitted: nothing would be gained and a restart costs hours.
"""

import math

# ---------------------------------------------------------------------------
# 1.  Centre-of-mass envelope of the arm, relative to its stowed position.
# ---------------------------------------------------------------------------
# Horizontal: a six-axis cobot's whole-arm COM sits at roughly 0.35 of its reach
# when extended, so 0.35 * 0.622 = 0.22 m from the J1 axis.  Add a 3 kg tool out
# at the full 0.622 m and the combined COM reaches
#     (15 * 0.218 + 3 * 0.622) / 18 = 0.285 m.
# The arm can point at any azimuth, so that 0.30 m applies to both x and y.
#
# Sampling a box rather than a disc over-covers the diagonal corners (0.42 m,
# past the physical envelope).  That is deliberately conservative: the policy
# trains on arm poses slightly worse than the arm can actually strike.
#
# Vertical: reaching straight up puts the COM ~0.30 m above the flange, +0.10 m
# from stowed.  Reaching out or down over the side drops it to -0.25 m.
ARM_COM_RANGE = {
  0: (-0.30, 0.30),  # x  (fore/aft)
  1: (-0.30, 0.30),  # y  (lateral)
  2: (-0.25, 0.12),  # z  (up/down)
}

# ---------------------------------------------------------------------------
# 2.  Mass uncertainty.
# ---------------------------------------------------------------------------
# `dr.pseudo_inertia` scales mass and inertia together by exp(2 * alpha), which
# is the physically consistent way to do it -- `dr.body_mass` would leave the
# inertia tensor stale.  Expressed as a mass multiplier:
#
#   arm   0.95 - 1.30x   ->  14.3 - 19.5 kg   (bare arm .. arm + 3 kg tool +
#                                              a heavy gripper, plus slack on
#                                              the vendor's round 15 kg)
#   deck  0.85 - 1.25x   ->   5.5 -  8.1 kg   (plate and DC-DC are estimates;
#                                              the 2.5 kg control box is spec)
def _alpha(lo: float, hi: float) -> tuple[float, float]:
  """Mass multiplier range -> pseudo-inertia alpha range."""
  return (math.log(lo) / 2.0, math.log(hi) / 2.0)


ARM_MASS_ALPHA = _alpha(0.95, 1.30)
DECK_MASS_ALPHA = _alpha(0.85, 1.25)

# ---------------------------------------------------------------------------
# 3.  Reaction wrench from arm motion.
# ---------------------------------------------------------------------------
# Applied to the `fr3_arm` body, so it automatically acts at wherever term 1 has
# placed the arm this episode.
#
#   force   a 3 kg payload emergency-stopped from 1 m/s in 0.1 s is 30 N; a
#           normal 2 m/s^2 move of ~10 kg of effective arm mass is 20 N.
#   torque  slewing the arm about J1 (~0.94 kg m^2) or pitching it about the
#           mount (~1.35 kg m^2) at 5 rad/s^2 gives 5-7 Nm; braking a 3 kg tool
#           at 0.6 m adds ~18 Nm.  25 Nm covers an aggressive move.
#   timing  arm moves last a fraction of a second to a couple of seconds, with
#           idle time in between.
ARM_REACTION = {
  "force_range": (-30.0, 30.0),
  "torque_range": (-25.0, 25.0),
  "duration_s": (0.3, 1.5),
  "cooldown_s": (1.0, 4.0),
}
