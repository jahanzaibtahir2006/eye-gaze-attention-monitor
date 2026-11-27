import numpy as np
import pytest

from gaze_monitor.attention import ALERT, OFF, ON, AttentionState, Calibration, Calibrator
from gaze_monitor.geometry import pitchyaw_to_vector

AXIS = [0.0, 0.0, -1.0]


def away(deg):
    return pitchyaw_to_vector(0.0, np.radians(deg))


def run(st, directions, fps=30.0, t0=0.0, head=None):
    states = []
    for i, d in enumerate(directions):
        states.append(st.update(d, t0 + i / fps, head))
    return states


def test_on_task_stays_on():
    st = AttentionState(Calibration(AXIS, None, 0.0, 30))
    states = run(st, [away(5)] * 120)
    assert set(states) == {ON}
    assert st.alert_count == 0


def test_looking_away_goes_off_then_alert_after_dwell():
    st = AttentionState(Calibration(AXIS, None, 0.0, 30), alert_after=2.0, grace=0.3)
    states = run(st, [away(45)] * 120)            # 4 s away
    first_off = states.index(OFF) / 30.0
    first_alert = states.index(ALERT) / 30.0
    assert 0.2 <= first_off <= 0.6
    assert 2.2 <= first_alert <= 2.6              # grace + dwell
    assert st.alert_count == 1


def test_single_lucky_frame_does_not_reset_timer():
    """Re-entering needs several consecutive inside frames (asymmetric hysteresis)."""
    st = AttentionState(Calibration(AXIS, None, 0.0, 30), smoothing=1.0)
    seq = [away(45)] * 45 + [away(0)] + [away(45)] * 45
    states = run(st, seq)
    assert ALERT in states


def test_returning_clears_alert():
    st = AttentionState(Calibration(AXIS, None, 0.0, 30), smoothing=1.0)
    states = run(st, [away(45)] * 90 + [away(0)] * 10)
    assert states[-1] == ON
    assert st.away_for == 0.0


def test_lost_face_counts_as_away():
    st = AttentionState(Calibration(AXIS, None, 0.0, 30))
    states = run(st, [None] * 90)
    assert states[-1] == ALERT


def test_head_turn_alone_triggers():
    head_axis = [0.0, 0.0, -1.0]
    st = AttentionState(Calibration(AXIS, head_axis, 0.0, 30), head_limit_deg=15.0)
    states = run(st, [away(0)] * 90, head=away(40))
    assert states[-1] == ALERT


def test_calibrator_mean_and_spread():
    cal = Calibrator(seconds=1.0)
    cal.start(0.0)
    for k in range(30):
        cal.add(away(10 if k % 2 else -10), away(0))
    c = cal.finish()
    assert c is not None
    assert np.allclose(c.axis, AXIS, atol=1e-6)
    assert c.spread_deg == pytest.approx(10.0, abs=0.01)


def test_calibrator_needs_samples():
    cal = Calibrator(seconds=1.0)
    cal.start(0.0)
    cal.add(None, None)
    assert cal.finish() is None


def test_calibration_roundtrip(tmp_path):
    c = Calibration([0.1, 0.2, -0.97], [0.0, 0.0, -1.0], 1.4, 80)
    p = tmp_path / "c.json"
    c.save(str(p))
    assert Calibration.load(str(p)) == c
