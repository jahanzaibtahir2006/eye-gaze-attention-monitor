"""One-Euro filter: jitter removal that stays responsive.

A plain moving average trades jitter for lag. The One-Euro filter (Casiez, Roussel and
Vogel, CHI 2012) adapts its cutoff to the signal's speed: heavy smoothing while the
gaze is still (where EyeGaze's ~15 degree frame-to-frame noise shows as flicker), light
smoothing during a real glance, so the arrow still snaps to where you look.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """Filters a numpy vector (any shape) sampled at irregular times."""

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.6, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        self._t = t
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = _alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()


class DirectionFilter(OneEuro):
    """One-Euro on a unit 3D direction, renormalised after each step."""

    def __call__(self, v, t: float) -> np.ndarray:
        out = super().__call__(v, t)
        n = np.linalg.norm(out)
        return out / n if n > 1e-9 else out
