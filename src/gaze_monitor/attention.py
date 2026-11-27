"""Attention state machine and calibration.

Off task means the gaze left its cone around the calibrated direction OR the head
turned past its limit. Each cue covers the other's blind spot: EyeGaze under-reads
large angles (about half the true angle) and barely moves when eyes look down at a
phone, while the head pose cannot be hidden by someone who really turns away.

Hardening a noisy signal (per the reference implementation's measurements):
- the direction is smoothed (EMA) before it is compared with the threshold;
- hysteresis is asymmetric: leaving needs a grace period, coming back needs several
  consecutive frames inside a tighter threshold;
- the dwell timer uses the caller's clock (wall time live, media time for files).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from .geometry import angle_between

ON, OFF, ALERT = "ON", "OFF", "ALERT"
LABELS = {ON: "ON TASK", OFF: "LOOKING AWAY", ALERT: "ATTENTION LOST"}


@dataclass
class Calibration:
    axis: list[float]
    head_axis: list[float] | None
    spread_deg: float
    samples: int

    @classmethod
    def default(cls) -> "Calibration":
        """Straight at the camera: the sensible guess for a screen-mounted webcam."""
        return cls(axis=[0.0, 0.0, -1.0], head_axis=None, spread_deg=0.0, samples=0)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(axis=d["axis"], head_axis=d.get("head_axis"),
                   spread_deg=float(d.get("spread_deg", 0.0)),
                   samples=int(d.get("samples", 0)))


class Calibrator:
    """Collects gaze and head directions for a few seconds, without blocking the UI."""

    def __init__(self, seconds: float = 3.0, min_samples: int = 5):
        self.seconds = seconds
        self.min_samples = min_samples
        self.t0: float | None = None
        self.gaze: list[np.ndarray] = []
        self.head: list[np.ndarray] = []

    def start(self, now: float) -> None:
        self.t0 = now
        self.gaze.clear()
        self.head.clear()

    @property
    def active(self) -> bool:
        return self.t0 is not None

    def remaining(self, now: float) -> float:
        return 0.0 if self.t0 is None else max(0.0, self.seconds - (now - self.t0))

    def add(self, direction, head) -> None:
        if direction is not None:
            self.gaze.append(np.asarray(direction, dtype=np.float64))
            if head is not None:
                self.head.append(np.asarray(head, dtype=np.float64))

    def done(self, now: float) -> bool:
        return self.t0 is not None and now - self.t0 >= self.seconds

    def finish(self) -> Calibration | None:
        self.t0 = None
        if len(self.gaze) < self.min_samples:
            return None
        axis = np.mean(self.gaze, axis=0)
        axis /= np.linalg.norm(axis)
        spread = float(np.mean([angle_between(g, axis) for g in self.gaze]))
        head_axis = None
        if len(self.head) >= self.min_samples:
            h = np.mean(self.head, axis=0)
            head_axis = (h / np.linalg.norm(h)).tolist()
        return Calibration(axis=axis.tolist(), head_axis=head_axis,
                           spread_deg=spread, samples=len(self.gaze))


class AttentionState:
    def __init__(self, calib: Calibration, cone_deg: float = 20.0,
                 head_limit_deg: float = 15.0, alert_after: float = 2.0,
                 grace: float = 0.3, smoothing: float = 0.35, clear_frames: int = 3,
                 exit_margin: float = 5.0):
        self.axis = np.asarray(calib.axis, dtype=np.float32)
        self.head_axis = (None if calib.head_axis is None
                          else np.asarray(calib.head_axis, dtype=np.float32))
        self.cone_deg = cone_deg
        self.head_limit_deg = head_limit_deg
        self.alert_after = alert_after
        self.grace = grace
        self.smoothing = smoothing
        self.clear_frames = clear_frames
        self.exit_margin = exit_margin

        self.state = ON
        self.off_since: float | None = None
        self.away_for = 0.0
        self.last_angle: float | None = None
        self.last_head_angle: float | None = None
        self._smoothed = None
        self._smoothed_head = None
        self._inside_run = 0

        # session statistics
        self.alert_count = 0
        self.total_away = 0.0
        self._last_now: float | None = None

    @property
    def margin(self) -> float:
        """While off task, the thresholds to get back in are tighter."""
        return 0.0 if self.off_since is None else self.exit_margin

    def _smooth(self, prev, d):
        if prev is None:
            return d
        v = (1 - self.smoothing) * prev + self.smoothing * d
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v

    def update(self, direction, now: float, head=None) -> str:
        if direction is None:
            # a lost face is not evidence of attention; keep the timer running
            self.last_angle = None
            self._inside_run = 0
            inside = False
        else:
            self._smoothed = self._smooth(self._smoothed,
                                          np.asarray(direction, dtype=np.float32))
            self.last_angle = angle_between(self._smoothed, self.axis)
            margin = self.margin
            inside = self.last_angle <= self.cone_deg - margin
            if self.head_axis is not None and head is not None:
                self._smoothed_head = self._smooth(self._smoothed_head,
                                                   np.asarray(head, dtype=np.float32))
                self.last_head_angle = angle_between(self._smoothed_head, self.head_axis)
                inside = inside and self.last_head_angle <= self.head_limit_deg - margin
            self._inside_run = self._inside_run + 1 if inside else 0

        prev_state = self.state
        if inside and self._inside_run >= self.clear_frames:
            self.off_since = None
            self.away_for = 0.0
            self.state = ON
        elif inside and self.off_since is None:
            self.state = ON
        else:
            if self.off_since is None:
                self.off_since = now
            self.away_for = max(0.0, now - self.off_since - self.grace)
            if self.away_for >= self.alert_after:
                self.state = ALERT
            elif now - self.off_since >= self.grace:
                self.state = OFF
            else:
                self.state = ON

        if self.state == ALERT and prev_state != ALERT:
            self.alert_count += 1
        if self._last_now is not None and self.state != ON:
            self.total_away += max(0.0, now - self._last_now)
        self._last_now = now
        return self.state
