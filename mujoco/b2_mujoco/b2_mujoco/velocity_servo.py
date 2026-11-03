"""Measured velocity feedback for the simulated joint-policy actuator.

This compensates the exported policy's tracking bias, below the navigation and
safety stack. It changes neither the requested chassis velocity nor odometry.
Only the MuJoCo launch enables it; the standalone policy runner stays unchanged.
"""

import math

import numpy as np


class VelocityServo:
    def __init__(self, lower, upper, *, kp=.7, ki=.5, tau=.25):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.kp, self.ki, self.tau = kp, ki, tau
        self.filtered = np.zeros(3)
        self.integral = np.zeros(3)
        self.initialized = False

    def reset(self):
        self.integral[:] = 0.
        self.filtered[:] = 0.
        self.initialized = False

    def step(self, requested, measured, dt):
        """Return policy input and gait intent; missing feedback stops motion.

        The 0.01 per-axis deadband matches the project's physical B2 SDK bridge.
        A zero/expired request clears accumulated correction immediately.
        """
        requested = np.asarray(requested, dtype=float)
        if (measured is None or not np.isfinite(requested).all()
                or not np.isfinite(measured).all() or not math.isfinite(dt) or dt <= 0
                or np.max(np.abs(requested)) < .01):
            self.reset()
            return np.zeros(3), False
        measured = np.asarray(measured, dtype=float)
        if not self.initialized:
            self.filtered[:] = measured
            self.initialized = True
        self.filtered += -math.expm1(-dt / self.tau) * (measured - self.filtered)
        error = requested - self.filtered
        candidate = np.clip(self.integral + self.ki * dt * error, -.25, .25)
        raw = requested + np.clip(self.kp * error + candidate, -.35, .35)
        bounded = np.clip(raw, self.lower, self.upper)
        # Do not integrate farther into the trained velocity limits.
        integrate = (raw == bounded) | ((raw - bounded) * error <= 0.)
        self.integral = np.where(integrate, candidate, self.integral)
        drive = requested + np.clip(self.kp * error + self.integral, -.35, .35)
        return np.clip(drive, self.lower, self.upper), True
