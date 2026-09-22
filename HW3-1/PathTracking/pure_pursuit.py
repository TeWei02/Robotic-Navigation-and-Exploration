"""Deterministic pure-pursuit baseline for the path-tracking environment.

The course assignment trains a PPO policy to output the yaw-rate command of the
unicycle ("basic") kinematic model.  This module provides a classical
geometry-based baseline that consumes the same reference path and produces the
same control quantity, so the two can be compared directly:

    * ``set_path(path)``  receives an ``(N, 4)`` array of ``(x, y, yaw, curvature)``
      samples, the same layout produced by ``cubic_spline.cubic_spline_2d``.
    * ``command(pose)``   receives the current ``(x, y, yaw)`` pose in degrees and
      returns the ``(v, w)`` pair of the basic control type, where ``w`` is a yaw
      rate in degrees per second.

Pure pursuit picks a look-ahead point on the path, measures the heading error to
that point and converts it into a curvature request, ``kappa = 2 sin(alpha) / Ld``,
which becomes a yaw rate through ``w = kappa * v``.
"""

import numpy as np

from PathTracking.controller import Controller


class PurePursuitController(Controller):
    """Geometry-based path-tracking baseline with a fixed look-ahead distance."""

    def __init__(self, lookahead=40.0, v=12.0, w_range=60.0, goal_tol=10.0):
        # Look-ahead distance along the path, in the simulator's length unit.
        self.lookahead = float(lookahead)
        # Constant forward speed of the basic kinematic model.
        self.v = float(v)
        # Yaw-rate saturation of the simulator, in degrees per second.
        self.w_range = float(w_range)
        # Goal tolerance used to decide whether the last path sample is targeted.
        self.goal_tol = float(goal_tol)
        self.path = None
        self.s = None

    def set_path(self, path):
        """Store the reference path and pre-compute its cumulative arc length."""
        self.path = np.asarray(path, dtype=float)
        if self.path.ndim != 2 or self.path.shape[1] < 2:
            raise ValueError("path must be an (N, >=2) array of (x, y, ...) samples")
        step = np.hypot(np.diff(self.path[:, 0]), np.diff(self.path[:, 1]))
        self.s = np.concatenate(([0.0], np.cumsum(step)))

    def _lookahead_point(self, pose):
        """Return the path point that lies ``lookahead`` arc length ahead of the pose."""
        d2 = (self.path[:, 0] - pose[0]) ** 2 + (self.path[:, 1] - pose[1]) ** 2
        idx = int(np.argmin(d2))
        s_target = self.s[idx] + self.lookahead
        if s_target >= self.s[-1]:
            return self.path[-1, :2]
        j = int(np.searchsorted(self.s, s_target, side="right") - 1)
        j = min(max(j, 0), len(self.s) - 2)
        span = self.s[j + 1] - self.s[j]
        ratio = 0.0 if span <= 0 else (s_target - self.s[j]) / span
        return self.path[j, :2] + ratio * (self.path[j + 1, :2] - self.path[j, :2])

    def command(self, pose, v=None):
        """Return the ``(v, w)`` command for the basic kinematic model."""
        if self.path is None:
            raise RuntimeError("set_path() must be called before command()")
        speed = self.v if v is None else float(v)
        target = self._lookahead_point(pose)
        dx = target[0] - pose[0]
        dy = target[1] - pose[1]
        # Heading error between the vehicle and the look-ahead point.
        alpha = np.arctan2(dy, dx) - np.deg2rad(pose[2])
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        kappa = 2.0 * np.sin(alpha) / self.lookahead
        w = float(np.rad2deg(kappa * speed))
        w = float(np.clip(w, -self.w_range, self.w_range))
        return speed, w

    # The abstract base class declares a ``feedback`` hook; the basic kinematic
    # model consumes ``w`` directly, so ``command`` is the entry point used by
    # the training environment and by the headless reference driver.
    def feedback(self, info):
        pose = info["pose"] if isinstance(info, dict) else info
        return self.command(pose)
