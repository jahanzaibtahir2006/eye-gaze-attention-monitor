"""Rendering: the live monitor view, the four pipeline stages, and the studio layout.

Three views, switchable at runtime with `v`:

    studio    1920x1080: monitor + live timeline on the left, the four model stages
              stacked on the right. Made for screen recordings and demo videos.
    monitor   the camera frame with the attention overlay only.
    pipeline  1920x1080 2x2 grid of the four stages, each at large size.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .attention import ALERT, LABELS, OFF, ON, AttentionState
from .geometry import EYE_H, EYE_W, head_rotation
from .pipeline import FrameResult
from .text import put_text, text_size

# BGR palette
BG = (17, 17, 19)
PANEL = (27, 27, 31)
BORDER = (52, 52, 58)
WHITE = (245, 245, 245)
MUTED = (150, 150, 158)
DIM = (95, 95, 102)
SHADOW = (12, 12, 12)
GREEN = (96, 214, 92)
AMBER = (40, 186, 246)
RED = (72, 72, 236)
RIGHT_EYE = (236, 168, 46)     # blue: subject's right eye
LEFT_EYE = (96, 206, 96)       # green: subject's left eye
ORANGE = (58, 142, 246)
CYAN = (214, 196, 72)

STATE_COLOUR = {ON: GREEN, OFF: AMBER, ALERT: RED}

OUT_W, OUT_H = 1920, 1080


@dataclass
class HudInfo:
    fps: float
    device: str
    gpu_name: str
    timings: dict
    mode: str                     # "ready" | "calibrating" | "monitoring"
    calib_remaining: float = 0.0
    calib_spread: float | None = None
    session_seconds: float = 0.0
    recording: bool = False


# --------------------------------------------------------------------------- primitives


@lru_cache(maxsize=32)
def _solid_template(h: int, w: int, colour: tuple) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:, :] = np.array(colour, np.uint8)
    return img


def solid(h: int, w: int, colour) -> np.ndarray:
    """A filled image. np.full with a tuple fill costs ~3 ms per 1080p frame; a cached
    template copy costs ~0.3 ms, which matters at 30 FPS."""
    return _solid_template(int(h), int(w), tuple(int(c) for c in colour)).copy()


def blend_rect(img, p0, p1, colour, alpha):
    x0, y0 = max(int(p0[0]), 0), max(int(p0[1]), 0)
    x1, y1 = min(int(p1[0]), img.shape[1]), min(int(p1[1]), img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * (1 - alpha)
              + np.array(colour, np.float32) * alpha).astype(np.uint8)


def rounded_rect(img, p0, p1, colour, radius=10, thickness=-1):
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    r = int(max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), colour, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), colour, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, colour, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x0 + r, y0), (x1 - r, y0), colour, thickness, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y1), (x1 - r, y1), colour, thickness, cv2.LINE_AA)
        cv2.line(img, (x0, y0 + r), (x0, y1 - r), colour, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y0 + r), (x1, y1 - r), colour, thickness, cv2.LINE_AA)
        for (cx, cy), a in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270),
                            ((x0 + r, y1 - r), 90), ((x1 - r, y1 - r), 0)):
            cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, colour, thickness, cv2.LINE_AA)


def arrow(img, a, b, colour, thick, tip=0.25):
    """Arrow with a dark under-stroke so it reads on skin, wall and shirt alike."""
    cv2.arrowedLine(img, a, b, SHADOW, thick + 4, cv2.LINE_AA, tipLength=tip)
    cv2.arrowedLine(img, a, b, colour, thick, cv2.LINE_AA, tipLength=tip)


def dot(img, p, r, colour):
    cv2.circle(img, p, r + 1, SHADOW, -1, cv2.LINE_AA)
    cv2.circle(img, p, r, colour, -1, cv2.LINE_AA)


def brackets(img, p0, p1, colour, thick):
    (x0, y0), (x1, y1) = p0, p1
    L = max(6, int(0.2 * min(x1 - x0, y1 - y0)))
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        for t, c in ((thick + 3, SHADOW), (thick, colour)):
            cv2.line(img, (cx, cy), (cx + dx * L, cy), c, t, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * L), c, t, cv2.LINE_AA)


def head_box(img, centre, size, pose, colour=AMBER):
    """A 3D box around the head in the head's own frame: its turn and tilt ARE the pose."""
    R = head_rotation(pose)
    a = size / 2.0
    back = 0.8 * a
    corners = np.array([[x, y, z - back] for z in (a, -a) for y in (-a, a) for x in (-a, a)],
                       dtype=np.float64)
    cam = corners @ R.T
    depth = 4.0 * size

    def proj(p):
        k = depth / (depth + p[2])
        return int(round(centre[0] + p[0] * k)), int(round(centre[1] + p[1] * k))

    pts = [proj(p) for p in cam]
    front = [(0, 1), (1, 3), (3, 2), (2, 0)]
    rest = [(4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    over = img.copy()
    for i, j in rest:
        cv2.line(over, pts[i], pts[j], colour, 2, cv2.LINE_AA)
    cv2.addWeighted(over, 0.45, img, 0.55, 0, img)
    for t, c in ((6, SHADOW), (3, colour)):
        for i, j in front:
            cv2.line(img, pts[i], pts[j], c, t, cv2.LINE_AA)


# --------------------------------------------------------------------------- monitor view


def draw_eye(vis, eye, s, arrow_delta):
    """Iris ring from EyeGaze's own landmarks, plus the shared gaze arrow."""
    radius = max(7.0 * s, eye.iris_radius)
    colour = RIGHT_EYE if eye.side == "right" else LEFT_EYE
    cx, cy = int(round(eye.iris_centre[0])), int(round(eye.iris_centre[1]))
    cv2.circle(vis, (cx, cy), int(round(radius)) + 1, SHADOW, max(3, int(5 * s)), cv2.LINE_AA)
    cv2.circle(vis, (cx, cy), int(round(radius)), colour, max(2, int(3 * s)), cv2.LINE_AA)
    cv2.circle(vis, (cx, cy), max(2, int(radius * 0.22)), WHITE, -1, cv2.LINE_AA)
    if arrow_delta is None:
        return
    length = float(np.hypot(arrow_delta[0], arrow_delta[1]))
    if length < 1.5:
        return
    ux, uy = arrow_delta[0] / length, arrow_delta[1] / length
    sx, sy = cx + ux * radius, cy + uy * radius
    tip = (int(round(sx + ux * length)), int(round(sy + uy * length)))
    arrow(vis, (int(round(sx)), int(round(sy))), tip, WHITE, max(2, int(3.5 * s)),
          tip=min(0.4, 12.0 * s / max(length, 1.0)))


def _meter(img, x, y, w, h, value, limit, colour, s):
    """Horizontal gauge: fill to value, tick at the limit, scale to 2x the limit."""
    full = max(limit * 2.0, 1e-6)
    rounded_rect(img, (x, y), (x + w, y + h), (60, 60, 66), radius=h // 2)
    if value is not None:
        fw = int(w * min(1.0, value / full))
        if fw > h:
            rounded_rect(img, (x, y), (x + fw, y + h), colour, radius=h // 2)
    lx = x + int(w * limit / full)
    cv2.line(img, (lx, y - int(4 * s)), (lx, y + h + int(4 * s)), WHITE, max(1, int(2 * s)),
             cv2.LINE_AA)


FACE_TAGS = {ON: "Watching the screen", OFF: "Looking away", ALERT: "Attention lost"}


def face_tag(vis, box, state, hud: HudInfo, s: float) -> None:
    """A status pill above the face, so the verdict reads without looking at the panel."""
    if state is not None:
        text, colour = FACE_TAGS[state], STATE_COLOUR[state]
    elif hud.mode == "calibrating":
        text, colour = f"Calibrating  {hud.calib_remaining:0.1f}s", AMBER
    else:
        text, colour = "Press C to calibrate", MUTED
    x0, y0, x1, y1 = box
    size = max(14, int(22 * s))
    tw, th = text_size(text, size, "bold")
    dot_r = int(size * 0.32)
    pad_x, pad_y = int(14 * s), int(9 * s)
    pw = pad_x * 2 + dot_r * 2 + int(10 * s) + tw
    ph = th + pad_y * 2
    cx = (x0 + x1) // 2
    px0 = int(np.clip(cx - pw // 2, 4, vis.shape[1] - pw - 4))
    py1 = y0 - int(12 * s)
    if py1 - ph < 4:                        # no room above the face: put it below
        py1 = min(vis.shape[0] - 4, y1 + int(12 * s) + ph)
    py0 = py1 - ph
    blend_rect(vis, (px0, py0), (px0 + pw, py1), (14, 14, 16), 0.82)
    rounded_rect(vis, (px0, py0), (px0 + pw, py1), colour, ph // 2, thickness=max(1, int(2 * s)))
    cy = (py0 + py1) // 2
    cv2.circle(vis, (px0 + pad_x + dot_r, cy), dot_r, colour, -1, cv2.LINE_AA)
    put_text(vis, text, (px0 + pad_x + dot_r * 2 + int(10 * s), cy), size, colour, "bold", "lm")


def draw_monitor(frame: np.ndarray, res: FrameResult | None, st: AttentionState,
                 hud: HudInfo, show_hud: bool = True, show_banner: bool = True) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]
    s = max(w / 1280.0, h / 720.0)
    state = st.state if hud.mode == "monitoring" else None
    colour = STATE_COLOUR.get(state, MUTED) if state else (AMBER if hud.mode == "calibrating" else MUTED)

    if res is not None:
        x0, y0, x1, y1 = (int(v) for v in res.box)
        brackets(vis, (x0, y0), (x1, y1), colour, max(2, int(3 * s)))
        # Before calibration the arrow would only show the person's fixed bias (a
        # third of the model's error), so it points off-centre even when looking
        # straight. Show it only once it is measured against the calibrated axis.
        arrows = res.arrow_delta if hud.mode == "monitoring" else None
        for eye in res.eyes:
            draw_eye(vis, eye, s, arrows)
        face_tag(vis, (x0, y0, x1, y1), state, hud, s)

    # alert frame: a pulsing red border the viewer cannot miss
    if state == ALERT:
        pulse = 0.55 + 0.45 * math.sin(time.monotonic() * 8.0)
        t = max(6, int(14 * s))
        over = vis.copy()
        cv2.rectangle(over, (0, 0), (w - 1, h - 1), RED, t * 2)
        cv2.addWeighted(over, pulse, vis, 1 - pulse, 0, vis)

    if not show_banner:
        return vis

    # ------------------------------------------------------------- top banner
    bh = int(92 * s)
    blend_rect(vis, (0, 0), (w, bh), (14, 14, 16), 0.78)
    pad = int(22 * s)

    if hud.mode == "ready":
        title, sub = "READY", ("Look at your screen and press C to calibrate"
                               if res is not None else "No face detected - move into view")
        put_text(vis, title, (pad, bh // 2), int(40 * s), MUTED, "bold", "lm")
        tw = text_size(title, int(40 * s), "bold")[0]
        put_text(vis, sub, (pad + tw + int(28 * s), bh // 2), int(22 * s),
                 AMBER if res is not None else RED, "semibold", "lm")
    elif hud.mode == "calibrating":
        title = "CALIBRATING"
        put_text(vis, title, (pad, bh // 2), int(40 * s), AMBER, "bold", "lm")
        tw = text_size(title, int(40 * s), "bold")[0]
        put_text(vis, f"Keep looking at the screen   {hud.calib_remaining:0.1f}s",
                 (pad + tw + int(28 * s), bh // 2), int(22 * s), WHITE, "semibold", "lm")
        bx = pad + tw + int(28 * s)
        frac = 1.0 - hud.calib_remaining / 3.0 if hud.calib_remaining else 1.0
        bw = int(w - bx - pad)
        y = bh - int(16 * s)
        cv2.rectangle(vis, (bx, y), (bx + bw, y + int(5 * s)), (60, 60, 66), -1)
        cv2.rectangle(vis, (bx, y), (bx + int(bw * max(0.0, min(1.0, frac))), y + int(5 * s)),
                      AMBER, -1)
    else:
        label = LABELS[st.state]
        title_size = int(40 * s)
        put_text(vis, label, (pad, bh // 2), title_size, colour, "bold", "lm")
        col_x = pad + text_size(LABELS[OFF], title_size, "bold")[0] + int(40 * s)

        m = st.margin
        gw = int(170 * s)
        rows = (("GAZE", st.last_angle, st.cone_deg - m),
                ("HEAD", st.last_head_angle, st.head_limit_deg - m))
        for r, (name, val, lim) in enumerate(rows):
            y = int(26 * s) + r * int(36 * s)
            put_text(vis, name, (col_x, y), int(16 * s), MUTED, "semibold", "lm")
            mx = col_x + int(58 * s)
            meter_colour = GREEN if (val is not None and val <= lim) else AMBER
            if val is None:
                meter_colour = DIM
            _meter(vis, mx, y - int(5 * s), gw, int(10 * s), val, lim, meter_colour, s)
            txt = "--" if val is None else f"{val:4.0f}° / {lim:.0f}°"
            put_text(vis, txt, (mx + gw + int(14 * s), y), int(18 * s), WHITE, "semibold", "lm")

        # dwell timer
        dx = col_x + int(58 * s) + gw + int(150 * s)
        if st.state in (OFF, ALERT) and dx < w - int(140 * s):
            remain = max(0.0, st.alert_after - st.away_for)
            frac = min(1.0, st.away_for / st.alert_after) if st.alert_after > 0 else 1.0
            bw = min(int(260 * s), w - dx - int(110 * s))
            y = bh // 2 - int(7 * s)
            rounded_rect(vis, (dx, y), (dx + bw, y + int(14 * s)), (60, 60, 66), int(7 * s))
            if frac > 0.03:
                rounded_rect(vis, (dx, y), (dx + int(bw * frac), y + int(14 * s)), colour,
                             int(7 * s))
            put_text(vis, "ALERT" if st.state == ALERT else f"{remain:3.1f}s",
                     (dx + bw + int(12 * s), bh // 2), int(20 * s), colour, "bold", "lm")

    if hud.recording:
        rx = w - pad - int(8 * s)
        if int(time.monotonic() * 2) % 2 == 0:
            cv2.circle(vis, (rx, int(18 * s)), int(7 * s), RED, -1, cv2.LINE_AA)
        put_text(vis, "REC", (rx - int(14 * s), int(18 * s)), int(15 * s), WHITE, "bold", "rm")

    if show_hud:
        line = f"{hud.fps:4.1f} FPS   {hud.device}"
        if hud.timings:
            line += "   " + "   ".join(f"{k} {v:.1f} ms" for k, v in hud.timings.items())
        tw, th = text_size(line, int(16 * s), "mono")
        blend_rect(vis, (0, h - th - int(22 * s)), (tw + int(36 * s), h), (14, 14, 16), 0.7)
        put_text(vis, line, (int(18 * s), h - int(11 * s)), int(16 * s), WHITE, "mono", "lb")
    return vis


# --------------------------------------------------------------------------- timeline


class Timeline:
    """Rolling chart of gaze and head angle against their thresholds, shaded by state."""

    def __init__(self, seconds: float = 20.0):
        self.seconds = seconds
        self.samples: deque = deque()

    def clear(self):
        self.samples.clear()

    def add(self, t, gaze, head, state):
        self.samples.append((t, gaze, head, state))
        while self.samples and t - self.samples[0][0] > self.seconds:
            self.samples.popleft()

    def draw(self, w, h, st: AttentionState | None) -> np.ndarray:
        img = solid(h, w, PANEL)
        pad_l, pad_r, pad_t, pad_b = 64, 24, 46, 34
        put_text(img, "ATTENTION TIMELINE", (20, 14), 17, MUTED, "semibold")
        lx = w - pad_r
        for name, col in (("head", AMBER), ("gaze", CYAN)):
            tw = text_size(name, 15, "semibold")[0]
            put_text(img, name, (lx - tw, 16), 15, WHITE, "semibold")
            cv2.line(img, (lx - tw - 34, 25), (lx - tw - 10, 25), col, 3, cv2.LINE_AA)
            lx -= tw + 54

        x0, y0, x1, y1 = pad_l, pad_t, w - pad_r, h - pad_b
        ymax = 60.0

        def ty(v):
            return int(y1 - (min(v, ymax) / ymax) * (y1 - y0))

        for v in (0, 20, 40, 60):
            y = ty(v)
            cv2.line(img, (x0, y), (x1, y), BORDER, 1)
            put_text(img, f"{v}°", (x0 - 12, y), 14, DIM, "regular", "rm")

        if not self.samples:
            put_text(img, "waiting for calibration", ((x0 + x1) // 2, (y0 + y1) // 2), 18,
                     DIM, "regular", "cm")
            return img
        t_end = self.samples[-1][0]

        def tx(t):
            return int(x1 - (t_end - t) / self.seconds * (x1 - x0))

        # state band under the plot
        prev = None
        for t, _, _, state in self.samples:
            x = tx(t)
            if prev is not None:
                cv2.rectangle(img, (prev[0], y1 + 6), (x, y1 + 14), STATE_COLOUR[prev[1]], -1)
            prev = (x, state)

        if st is not None:
            for lim, col in ((st.cone_deg, CYAN), (st.head_limit_deg, AMBER)):
                y = ty(lim)
                for xx in range(x0, x1, 14):
                    cv2.line(img, (xx, y), (min(xx + 7, x1), y), col, 1, cv2.LINE_AA)

        for idx, col in ((2, AMBER), (1, CYAN)):
            pts = [(tx(smp[0]), ty(smp[idx])) for smp in self.samples if smp[idx] is not None]
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)

        put_text(img, f"-{self.seconds:.0f}s", (x0, h - 14), 14, DIM, "regular", "lm")
        put_text(img, "now", (x1, h - 14), 14, DIM, "regular", "rm")
        return img


# --------------------------------------------------------------------------- stage tiles


class Framer:
    """A crop window that follows the face smoothly, so panels do not jitter."""

    def __init__(self, smoothing: float = 0.15):
        self.k = smoothing
        self.rect = None

    def update(self, target):
        t = np.asarray(target, np.float64)
        self.rect = t if self.rect is None else self.rect + self.k * (t - self.rect)
        return tuple(self.rect)


def fit_aspect(x0, y0, x1, y1, aspect, fw, fh):
    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    if bw / max(bh, 1e-6) < aspect:
        bw = bh * aspect
    else:
        bh = bw / aspect
    if bh > fh:
        bh, bw = fh, fh * aspect
    if bw > fw:
        bw, bh = fw, fw / aspect
    x0 = float(np.clip(cx - bw / 2, 0, fw - bw))
    y0 = float(np.clip(cy - bh / 2, 0, fh - bh))
    return x0, y0, x0 + bw, y0 + bh


def view(frame, rect, size):
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    x1, y1 = max(x1, x0 + 2), max(y1, y0 + 2)
    tile = cv2.resize(frame[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR)
    sx, sy = size[0] / (x1 - x0), size[1] / (y1 - y0)

    def at(p):
        return int(round((p[0] - x0) * sx)), int(round((p[1] - y0) * sy))
    return tile, at, sx


def stage_label(img, n, text, scale=1.0):
    size = int(20 * scale)
    num = str(n)
    nw = text_size(num, size, "bold")[0]
    tw = text_size(text, size, "semibold")[0]
    pad, gap = int(12 * scale), int(10 * scale)
    x0, y0 = int(12 * scale), int(12 * scale)
    hgt = int(size * 1.7)
    blend_rect(img, (x0, y0), (x0 + pad + nw + gap + tw + pad, y0 + hgt), (10, 10, 12), 0.75)
    put_text(img, num, (x0 + pad, y0 + hgt // 2), size, ORANGE, "bold", "lm")
    put_text(img, text, (x0 + pad + nw + gap, y0 + hgt // 2), size, WHITE, "semibold", "lm")


STAGE_TITLES = ["Face detection", "68 landmarks + head pose", "Eye patches (model input)",
                "Gaze direction"]


class StageRenderer:
    """Draws the four model stages as live tiles of any size."""

    def __init__(self):
        self.scene = Framer(0.12)
        self.head = Framer(0.18)
        self.strip = Framer(0.25)
        self.last: FrameResult | None = None

    def render(self, frame, res: FrameResult | None, size, scale=1.0):
        """Returns four BGR tiles of `size` (w, h)."""
        if res is not None and len(res.eyes) == 2:
            self.last = res
        res = self.last
        cw, ch = size
        fh, fw = frame.shape[:2]
        if res is None:
            tiles = []
            for n in range(4):
                tile = solid(ch, cw, PANEL)
                stage_label(tile, n + 1, STAGE_TITLES[n], scale)
                put_text(tile, "no face", (cw // 2, ch // 2), int(20 * scale), DIM,
                         "regular", "cm")
                tiles.append(tile)
            return tiles

        x0, y0, x1, y1 = res.box
        fx, fy, face_h = (x0 + x1) / 2, (y0 + y1) / 2, (y1 - y0)
        aspect = cw / ch
        ph = min(fh, 3.2 * face_h)
        scene = self.scene.update(fit_aspect(fx - ph * aspect / 2, fy - 0.4 * ph,
                                             fx + ph * aspect / 2, fy + 0.6 * ph, aspect, fw, fh))
        pad = 0.3 * (y1 - y0)
        head = self.head.update(fit_aspect(x0 - pad, y0 - pad, x1 + pad, y1 + pad, aspect, fw, fh))
        inter = float(np.linalg.norm(res.eyes[0].iris_centre - res.eyes[1].iris_centre))

        # 1 face detection
        p1, at, _ = view(frame, scene, size)
        brackets(p1, at((x0, y0)), at((x1, y1)), WHITE, max(2, int(3 * scale)))
        if res.landmark5 is not None:
            for p in res.landmark5:
                dot(p1, at(p), max(3, int(4 * scale)), ORANGE)
        put_text(p1, f"score {res.score:.2f}", (cw - int(14 * scale), ch - int(12 * scale)),
                 int(16 * scale), WHITE, "semibold", "rb")

        # 2 landmarks + head pose
        p2, at, s2 = view(frame, head, size)
        centre = np.mean([at(p) for p in res.landmarks], axis=0)
        head_box(p2, centre, 2.5 * inter * s2, res.pose)
        eye_idx = {i: (RIGHT_EYE if e.side == "right" else LEFT_EYE)
                   for e in res.eyes for i in e.landmark_idx}
        r = max(2, int(3 * scale))
        for i, p in enumerate(res.landmarks):
            dot(p2, at(p), r + (1 if i in eye_idx else 0), eye_idx.get(i, WHITE))
        pitch, yaw, roll = np.degrees(res.pose)
        put_text(p2, f"yaw {yaw:+.0f}°  pitch {pitch:+.0f}°  roll {roll:+.0f}°",
                 (cw - int(14 * scale), ch - int(12 * scale)), int(16 * scale), WHITE,
                 "semibold", "rb")

        # 3 eye patches: where they come from (top) and exactly what the model sees (bottom)
        p3 = solid(ch, cw, PANEL)
        g = int(12 * scale)
        pw = (cw - 3 * g) // 2
        ph_ = int(pw * EYE_H / EYE_W)
        if ph_ > ch * 0.55:
            ph_ = int(ch * 0.55)
            pw = int(ph_ * EYE_W / EYE_H)
        strip_h = ch - ph_ - 2 * g
        quads = np.concatenate([e.quad for e in res.eyes])
        qx0, qy0 = quads.min(axis=0)
        qx1, qy1 = quads.max(axis=0)
        qp = 0.1 * (qx1 - qx0)
        strip = self.strip.update(fit_aspect(qx0 - qp, qy0 - qp, qx1 + qp, qy1 + qp,
                                             cw / max(strip_h, 1), fw, fh))
        top, at, _ = view(frame, strip, (cw, strip_h))
        for e in res.eyes:
            col = RIGHT_EYE if e.side == "right" else LEFT_EYE
            q = np.array([at(p) for p in e.quad], np.int32)
            cv2.polylines(top, [q], True, SHADOW, 6, cv2.LINE_AA)
            cv2.polylines(top, [q], True, col, 3, cv2.LINE_AA)
        p3[:strip_h] = top
        total = 2 * pw + g
        start = (cw - total) // 2
        for k, e in enumerate(res.eyes):
            col = RIGHT_EYE if e.side == "right" else LEFT_EYE
            px, py = start + k * (pw + g), strip_h + g
            tile = cv2.resize(cv2.cvtColor(e.patch, cv2.COLOR_GRAY2BGR), (pw, ph_),
                              interpolation=cv2.INTER_NEAREST)
            p3[py:py + ph_, px:px + pw] = tile
            cv2.rectangle(p3, (px - 2, py - 2), (px + pw + 1, py + ph_ + 1), col, 3)

        # 4 gaze: per-eye estimates (colour) and their mean (white)
        p4, at, s4 = view(frame, head, size)
        inter4 = inter * s4
        for e in res.eyes:
            col = RIGHT_EYE if e.side == "right" else LEFT_EYE
            c = at(e.iris_centre)
            rr = max(6, int(e.iris_radius * s4))
            cv2.circle(p4, c, rr + 2, SHADOW, 5, cv2.LINE_AA)
            cv2.circle(p4, c, rr, col, 3, cv2.LINE_AA)
            v = e.display_gaze
            arrow(p4, c, (int(c[0] + v[0] * 1.6 * inter4),
                          int(c[1] + v[1] * 1.6 * inter4)), col, max(2, int(3 * scale)))
        mid = np.mean([at(e.iris_centre) for e in res.eyes], axis=0).astype(int)
        mv = res.display_direction
        arrow(p4, (int(mid[0]), int(mid[1])),
              (int(mid[0] + mv[0] * 2.2 * inter4), int(mid[1] + mv[1] * 2.2 * inter4)),
              WHITE, max(3, int(5 * scale)), tip=0.22)

        tiles = [p1, p2, p3, p4]
        for n, tile in enumerate(tiles, start=1):
            stage_label(tile, n, STAGE_TITLES[n - 1], scale)
        return tiles


# --------------------------------------------------------------------------- layouts


def _header(canvas, hud: HudInfo, st: AttentionState | None, title: str,
            show_stats: bool = True):
    hh = 64
    canvas[:hh] = (12, 12, 14)
    cv2.line(canvas, (0, hh - 1), (OUT_W, hh - 1), BORDER, 1)
    put_text(canvas, title, (28, hh // 2), 26, WHITE, "bold", "lm")
    tw = text_size(title, 26, "bold")[0]
    put_text(canvas, "Face detection  →  68 landmarks + head pose  →  EyeGaze x2",
             (28 + tw + 24, hh // 2 + 1), 17, MUTED, "regular", "lm")

    # right side: chips
    x = OUT_W - 28
    chips = [(f"{hud.fps:4.1f} FPS", WHITE), (hud.gpu_name or hud.device, GREEN)]
    if show_stats and st is not None and hud.mode == "monitoring":
        mins, secs = divmod(int(hud.session_seconds), 60)
        chips.insert(0, (f"alerts {st.alert_count}   away {st.total_away:4.1f}s   "
                         f"session {mins:02d}:{secs:02d}", MUTED))
    for text, col in chips:
        tw = text_size(text, 17, "semibold")[0]
        rounded_rect(canvas, (x - tw - 28, 14), (x, hh - 14), (34, 34, 40), 17)
        put_text(canvas, text, (x - 14, hh // 2), 17, col, "semibold", "rm")
        x -= tw + 40
    return hh


def _status_card(w, h, st: AttentionState, hud: HudInfo) -> np.ndarray:
    """Big state readout, the two cues against their limits, and session counters."""
    img = solid(h, w, PANEL)
    pad = 24
    if hud.mode == "monitoring":
        label, colour = LABELS[st.state], STATE_COLOUR[st.state]
    elif hud.mode == "calibrating":
        label, colour = "CALIBRATING", AMBER
    else:
        label, colour = "READY", MUTED
    put_text(img, "STATUS", (pad, 16), 17, MUTED, "semibold")
    rounded_rect(img, (pad, 46), (w - pad, 118), (20, 20, 23), 14)
    rounded_rect(img, (pad, 46), (w - pad, 118), colour, 14, thickness=2)
    cv2.circle(img, (pad + 34, 82), 11, colour, -1, cv2.LINE_AA)
    put_text(img, label, (pad + 60, 82), 34, colour, "bold", "lm")

    y = 146
    if hud.mode == "monitoring":
        m = st.margin
        for name, val, lim in (("Gaze off-axis", st.last_angle, st.cone_deg - m),
                               ("Head turn", st.last_head_angle, st.head_limit_deg - m)):
            put_text(img, name, (pad, y), 18, WHITE, "semibold")
            txt = "--" if val is None else f"{val:.0f}°  /  limit {lim:.0f}°"
            put_text(img, txt, (w - pad, y), 18, MUTED, "semibold", "rt")
            col = DIM if val is None else (GREEN if val <= lim else AMBER)
            _meter(img, pad, y + 34, w - 2 * pad, 12, val, lim, col, 1.0)
            y += 74
        # dwell
        frac = min(1.0, st.away_for / st.alert_after) if st.alert_after > 0 else 0.0
        put_text(img, "Time away", (pad, y), 18, WHITE, "semibold")
        put_text(img, f"{st.away_for:.1f}s  /  alert at {st.alert_after:.0f}s", (w - pad, y), 18,
                 MUTED, "semibold", "rt")
        rounded_rect(img, (pad, y + 34), (w - pad, y + 46), (60, 60, 66), 6)
        if frac > 0.02:
            rounded_rect(img, (pad, y + 34), (pad + int((w - 2 * pad) * frac), y + 46),
                         colour, 6)
        y += 80
        mins, secs = divmod(int(hud.session_seconds), 60)
        stats = (("ALERTS", f"{st.alert_count}"), ("AWAY", f"{st.total_away:.0f}s"),
                 ("SESSION", f"{mins:02d}:{secs:02d}"))
        cw = (w - 2 * pad - 2 * 12) // 3
        for i, (k, v) in enumerate(stats):
            x = pad + i * (cw + 12)
            rounded_rect(img, (x, y), (x + cw, y + 74), (34, 34, 40), 12)
            put_text(img, v, (x + cw // 2, y + 30), 28, WHITE, "bold", "cm")
            put_text(img, k, (x + cw // 2, y + 58), 13, MUTED, "semibold", "cm")
    elif hud.mode == "calibrating":
        put_text(img, "Keep looking at your screen.", (pad, y), 20, WHITE, "semibold")
        put_text(img, f"{hud.calib_remaining:0.1f} s", (pad, y + 40), 44, AMBER, "bold")
    else:
        for i, line in enumerate(("1. Look at your screen", "2. Press C to calibrate (3 s)",
                                  "3. Look away and watch the alert")):
            put_text(img, line, (pad, y + i * 40), 20, WHITE if i else AMBER, "semibold")

    return img


def compose_studio(frame, res, st, hud: HudInfo, stages: StageRenderer,
                   timeline: Timeline, title: str) -> np.ndarray:
    """Header; monitor (left) beside status + timeline (right); four stages below."""
    canvas = solid(OUT_H, OUT_W, BG)
    hh = _header(canvas, hud, st, title, show_stats=False)
    g = 12

    # bottom row: four stage tiles
    sh = 276
    sw = (OUT_W - 5 * g) // 4
    sy = OUT_H - g - sh
    tiles = stages.render(frame, res, (sw, sh), scale=0.72)
    for k, tile in enumerate(tiles):
        x = g + k * (sw + g)
        canvas[sy:sy + sh, x:x + sw] = tile
    lat = {0: "detect", 1: "landmarks", 3: "gaze"}
    for k, key in lat.items():
        if key in hud.timings:
            x = g + k * (sw + g)
            put_text(canvas, f"{hud.timings[key]:.1f} ms", (x + sw - 12, sy + 14), 15, WHITE,
                     "mono", "rt")

    # top row: monitor at 16:9, filling the height left above the tiles
    mh = sy - g - (hh + g)
    mw = int(round(mh * 16 / 9))
    fh, fw = frame.shape[:2]
    k = min(mw / fw, mh / fh)
    dw, dh = int(round(fw * k)), int(round(fh * k))
    mon_frame = frame if (dw, dh) == (fw, fh) else cv2.resize(
        frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
    res_m = _scale_result(res, k) if res is not None and k != 1.0 else res
    mon = draw_monitor(mon_frame, res_m, st, hud, show_hud=False, show_banner=False)
    my = hh + g
    canvas[my:my + mh, g:g + mw] = 0
    ox, oy = g + (mw - dw) // 2, my + (mh - dh) // 2
    canvas[oy:oy + dh, ox:ox + dw] = mon
    if hud.recording:
        if int(time.monotonic() * 2) % 2 == 0:
            cv2.circle(canvas, (g + mw - 70, my + 26), 8, RED, -1, cv2.LINE_AA)
        put_text(canvas, "REC", (g + mw - 18, my + 26), 16, WHITE, "bold", "rm")

    # right of the monitor: status card over timeline
    rx = g + mw + g
    rw = OUT_W - rx - g
    card_h = 440
    canvas[my:my + card_h, rx:rx + rw] = _status_card(rw, card_h, st, hud)
    ty = my + card_h + g
    th = sy - g - ty
    canvas[ty:ty + th, rx:rx + rw] = timeline.draw(rw, th,
                                                   st if hud.mode == "monitoring" else None)
    return canvas


def compose_pipeline(frame, res, st, hud: HudInfo, stages: StageRenderer,
                     title: str) -> np.ndarray:
    canvas = solid(OUT_H, OUT_W, BG)
    hh = _header(canvas, hud, st, title)
    g = 12
    cw = (OUT_W - 3 * g) // 2
    ch = (OUT_H - hh - 3 * g) // 2
    tiles = stages.render(frame, res, (cw, ch), scale=1.15)
    for k, tile in enumerate(tiles):
        cx = g + (k % 2) * (cw + g)
        cy = hh + g + (k // 2) * (ch + g)
        canvas[cy:cy + ch, cx:cx + cw] = tile
    # state pill in the centre
    if st is not None and hud.mode == "monitoring":
        label = LABELS[st.state]
        col = STATE_COLOUR[st.state]
        tw = text_size(label, 26, "bold")[0]
        cx, cy = OUT_W // 2, hh + g + ch + g // 2
        rounded_rect(canvas, (cx - tw // 2 - 26, cy - 26), (cx + tw // 2 + 26, cy + 26),
                     (14, 14, 16), 26)
        rounded_rect(canvas, (cx - tw // 2 - 26, cy - 26), (cx + tw // 2 + 26, cy + 26),
                     col, 26, thickness=2)
        put_text(canvas, label, (cx, cy), 26, col, "bold", "cm")
    return canvas


def _scale_result(res: FrameResult, k: float) -> FrameResult:
    """A shallow copy of a result with image-space geometry scaled by k (for display)."""
    from dataclasses import replace

    from .pipeline import Eye
    eyes = [Eye(side=e.side, patch=e.patch, quad=e.quad * k, iris_centre=e.iris_centre * k,
                iris_radius=e.iris_radius * k, gaze=e.gaze, display_gaze=e.display_gaze,
                landmark_idx=e.landmark_idx)
            for e in res.eyes]
    return replace(res, box=tuple(v * k for v in res.box),
                   landmark5=None if res.landmark5 is None else res.landmark5 * k,
                   landmarks=res.landmarks * k, eyes=eyes,
                   arrow_delta=None if res.arrow_delta is None else res.arrow_delta * k)
