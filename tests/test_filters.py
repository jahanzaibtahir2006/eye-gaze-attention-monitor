"""The arrow filter: much less flicker while still, and still quick on a real glance.

Noise model: EyeGaze's per-frame jitter, ~8 degrees per axis at 25 FPS.
"""

import numpy as np

from gaze_monitor.filters import DirectionFilter
from gaze_monitor.geometry import angle_between, pitchyaw_to_vector

FPS = 25
SIG = np.radians(8)


def _noisy(rng, yaw=0.0):
    return pitchyaw_to_vector(rng.normal(0, SIG), yaw + rng.normal(0, SIG))


def test_flicker_reduced_at_least_4x():
    rng = np.random.default_rng(0)
    f = DirectionFilter(min_cutoff=0.3, beta=0.5)
    raw, out = [], []
    prev_r = prev_o = None
    for i in range(200):
        r = _noisy(rng)
        o = f(r, i / FPS)
        if i > 30:
            raw.append(angle_between(r, prev_r))
            out.append(angle_between(o, prev_o))
        prev_r, prev_o = r, o
    assert np.mean(out) * 4 < np.mean(raw)


def test_real_glance_followed_within_half_a_second():
    rng = np.random.default_rng(1)
    f = DirectionFilter(min_cutoff=0.3, beta=0.5)
    for i in range(100):
        f(_noisy(rng), i / FPS)
    target = pitchyaw_to_vector(0.0, np.radians(30))
    errs = []
    for i in range(100, 100 + FPS):            # one second after a 30 degree glance
        v = f(_noisy(rng, np.radians(30)), i / FPS)
        errs.append(angle_between(v, target))
    assert np.mean(errs[FPS // 2:]) < 10       # settled by the second half-second


def test_output_stays_unit_length():
    rng = np.random.default_rng(2)
    f = DirectionFilter()
    for i in range(50):
        assert abs(np.linalg.norm(f(_noisy(rng), i / FPS)) - 1) < 1e-9
