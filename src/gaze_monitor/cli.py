"""Command-line entry point and the main video loop.

    gaze-monitor                          # webcam 0, studio view, GPU
    gaze-monitor --record                 # also save an H.264 MP4 in recordings/
    gaze-monitor --source clip.mp4 --calibrate --no-display --record out.mp4

Keys in the window:
    C  calibrate (look at your screen for 3 s)      V  switch view
    R  start/stop recording                         H  toggle stats line
    S  save a screenshot                            F  fullscreen
    O  rotate the phone picture 90 deg              Q / Esc  quit
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2

from . import __version__
from .alerts import Beeper
from .attention import ALERT, AttentionState, Calibration, Calibrator
from .overlay import (HudInfo, StageRenderer, Timeline, compose_pipeline, compose_studio,
                      draw_monitor)
from .pipeline import GazePipeline
from .video import Recorder, open_source

log = logging.getLogger("gaze_monitor")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIEWS = ("studio", "monitor", "pipeline")
WINDOW = "Eye Gaze Attention Monitor"


def gpu_name() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        name = out.stdout.strip().splitlines()[0].strip()
        return name.replace("NVIDIA ", "") or None
    except (OSError, IndexError, subprocess.SubprocessError):
        return None


def enable_dpi_awareness() -> None:
    """Stop Windows from bitmap-stretching the OpenCV window on 125%/150% displays.

    An unaware window is scaled up by the display factor, so a 1600x900 window becomes
    2000x1125 physical pixels, overflows a 1080p screen and looks blurry.
    """
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor aware
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def screen_work_area() -> tuple[int, int, int, int]:
    """(x, y, w, h) of the primary screen minus the taskbar, in physical pixels."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        rect = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    return 0, 0, 1920, 1040


def fit_window(window: str, aspect: float = 16 / 9, fill: float = 0.94) -> None:
    """Size the window to the largest 16:9 that fits the work area, and centre it."""
    x, y, w, h = screen_work_area()
    title_bar = 40
    ww = int(w * fill)
    wh = int(ww / aspect)
    if wh > (h - title_bar) * fill:
        wh = int((h - title_bar) * fill)
        ww = int(wh * aspect)
    cv2.resizeWindow(window, ww, wh)
    cv2.moveWindow(window, x + (w - ww) // 2, y + max(0, (h - wh - title_bar) // 2))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="gaze-monitor",
                                 description="Webcam eye-gaze attention monitor (GPU).")
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config.toml"),
                    help="TOML file with defaults for any option below")
    ap.add_argument("--source", default="0",
                    help="'phone' (Android over USB), a webcam index, or a video file")
    ap.add_argument("--phone-facing", choices=("front", "back"), default="front")
    ap.add_argument("--phone-size", type=int, default=1920,
                    help="longest side of the phone stream, pixels")
    ap.add_argument("--phone-rotate", type=int, choices=(0, 90, 180, 270), default=0)
    ap.add_argument("--scrcpy-dir", default=None,
                    help="scrcpy release folder (found automatically in Downloads)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--camera-fps", type=int, default=30)
    ap.add_argument("--models", default=os.path.join(PROJECT_ROOT, "models"))
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--view", choices=VIEWS, default="studio")
    ap.add_argument("--title", default="Eye Gaze Attention Monitor")
    ap.add_argument("--cone", type=float, default=20.0,
                    help="gaze cone half-angle around the calibrated direction (deg)")
    ap.add_argument("--head-limit", type=float, default=15.0,
                    help="max head turn from the calibrated pose (deg)")
    ap.add_argument("--alert-after", type=float, default=2.0,
                    help="seconds off task before the alert")
    ap.add_argument("--detect-every", type=int, default=1,
                    help="run the face detector every N frames")
    ap.add_argument("--calibrate", action="store_true",
                    help="calibrate immediately (files: on their first seconds)")
    ap.add_argument("--calib-seconds", type=float, default=3.0)
    ap.add_argument("--calib-file", default=os.path.join(PROJECT_ROOT, "calibration.json"))
    ap.add_argument("--load-calib", action="store_true",
                    help="start from the saved calibration instead of waiting for C")
    ap.add_argument("--record", nargs="?", const="auto", default=None,
                    help="save the rendered view as MP4 (optional path)")
    ap.add_argument("--record-fps", type=float, default=30.0)
    ap.add_argument("--csv", default=None, help="write a per-frame log to this CSV")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return ap


def parse_args(argv=None) -> argparse.Namespace:
    ap = build_parser()
    pre, _ = ap.parse_known_args(argv)
    valid = {a.dest for a in ap._actions}
    # config.toml, then config.local.toml next to it (personal, git-ignored) on top
    paths = [pre.config]
    if pre.config:
        paths.append(os.path.join(os.path.dirname(os.path.abspath(pre.config)), "config.local.toml"))
    for path in paths:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                cfg = tomllib.load(f)
            ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()
                               if k.replace("-", "_") in valid})
    return ap.parse_args(argv)


def auto_record_path() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(PROJECT_ROOT, "recordings", f"gaze-monitor-{stamp}.mp4")


class App:
    def __init__(self, args):
        self.args = args
        self.pipe = GazePipeline(args.models, device=args.device,
                                 detect_every=args.detect_every)
        log.info("warming up models on %s", self.pipe.device_label)
        self.pipe.warmup()
        self.source = open_source(args.source, args.width, args.height, args.camera_fps,
                                  phone=dict(scrcpy_dir=args.scrcpy_dir,
                                             facing=args.phone_facing,
                                             max_size=args.phone_size, fps=args.camera_fps,
                                             rotate=args.phone_rotate))
        self.live = self.source.is_live
        self.view = args.view
        self.show_hud = True
        self.stages = StageRenderer()
        self.timeline = Timeline(20.0)
        self.beeper = Beeper(enabled=not args.no_audio and not args.no_display)
        self.calibrator = Calibrator(args.calib_seconds)
        self.calib: Calibration | None = None
        self.st = AttentionState(Calibration.default(), args.cone, args.head_limit,
                                 args.alert_after)
        self.mode = "ready"
        self.recorder: Recorder | None = None
        self.gpu = gpu_name() if self.pipe.device_label.startswith("GPU") else None
        self.device_text = (f"{self.gpu} · CUDA" if self.gpu else self.pipe.device_label)
        self.session_t0: float | None = None
        self.csv_file = None
        self.csv_writer = None
        if args.csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
            self.csv_file = open(args.csv, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["t", "state", "gaze_deg", "head_deg", "face"])

        if args.load_calib and os.path.exists(args.calib_file):
            self.apply_calibration(Calibration.load(args.calib_file))
            log.info("loaded calibration from %s", args.calib_file)
        if args.record:
            self.start_recording(None if args.record == "auto" else args.record)

    # ------------------------------------------------------------------ helpers

    def apply_calibration(self, calib: Calibration) -> None:
        self.calib = calib
        self.st = AttentionState(calib, self.args.cone, self.args.head_limit,
                                 self.args.alert_after)
        self.pipe.set_reference(calib.axis)
        self.timeline.clear()
        self.mode = "monitoring"
        self.session_t0 = None

    def start_recording(self, path: str | None) -> None:
        path = path or auto_record_path()
        size = (1920, 1080) if self.view != "monitor" else self.source.size
        self.recorder = Recorder(path, size, self.args.record_fps)

    def stop_recording(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
            print(f"saved recording: {self.recorder.path}")
            self.recorder = None

    def hud(self, fps, now) -> HudInfo:
        """Snapshot of everything the renderer reads that is not frame data."""
        h = HudInfo(fps=fps, device=self.device_text, gpu_name=self.device_text,
                    timings=dict(self.pipe.timings), mode=self.mode,
                    calib_remaining=self.calibrator.remaining(now),
                    session_seconds=0.0 if self.session_t0 is None
                    else now - self.session_t0,
                    recording=self.recorder is not None)
        h.now = now
        return h

    def output(self, out, now, display) -> bool:
        """Record and show one rendered frame, and handle keys. Returns True to quit."""
        if self.recorder is not None:
            self.recorder.write(out, now)
        if not display:
            return False
        cv2.imshow(WINDOW, out)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return True
        if key == ord("c"):
            self.calibrator.start(now)
            self.mode = "calibrating"
        elif key == ord("v"):
            self.view = VIEWS[(VIEWS.index(self.view) + 1) % len(VIEWS)]
        elif key == ord("h"):
            self.show_hud = not self.show_hud
        elif key == ord("o") and hasattr(self.source, "cycle_rotation"):
            deg = self.source.cycle_rotation()
            self.pipe.reset()
            print(f"phone picture rotated to {deg} deg; press C to recalibrate")
        elif key == ord("r"):
            if self.recorder is None:
                self.start_recording(None)
            else:
                self.stop_recording()
        elif key == ord("s"):
            shot = os.path.join(PROJECT_ROOT, "recordings",
                                datetime.now().strftime("shot-%Y%m%d-%H%M%S.png"))
            os.makedirs(os.path.dirname(shot), exist_ok=True)
            cv2.imwrite(shot, out)
            print(f"saved screenshot: {shot}")
        elif key == ord("f"):
            full = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL if full == cv2.WINDOW_FULLSCREEN
                                  else cv2.WINDOW_FULLSCREEN)
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1

    def render(self, frame, res, hud: HudInfo):
        if self.view == "studio":
            return compose_studio(frame, res, self.st, hud, self.stages, self.timeline,
                                  self.args.title)
        if self.view == "pipeline":
            return compose_pipeline(frame, res, self.st, hud, self.stages, self.args.title)
        return draw_monitor(frame, res, self.st, hud, show_hud=self.show_hud)

    # ------------------------------------------------------------------ loop

    def run(self) -> int:
        args = self.args
        display = not args.no_display
        if display:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            fit_window(WINDOW)
            if args.fullscreen:
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print(__doc__.split("Keys in the window:")[1].rstrip())

        media_fps = self.source.fps or 30.0
        frame_i = 0
        fps_ema, prev = 0.0, time.perf_counter()
        rewound = False
        if args.calibrate:
            self.calibrator.start(0.0 if not self.live else time.monotonic())
            self.mode = "calibrating"

        # Rendering a 1080p view (~19 ms) runs on a worker thread while the main thread
        # runs the next frame's inference (~16 ms on the GPU); OpenCV, NumPy and ONNX
        # Runtime release the GIL, so the two genuinely overlap. All state is mutated on
        # the main thread only, and only while no render is in flight.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")
        pending = None
        stop = False
        try:
            while True:
                ok, frame = self.source.read()
                if not ok:
                    break
                now = time.monotonic() if self.live else frame_i / media_fps
                res = self.pipe(frame, now)

                if pending is not None:
                    out, out_now = pending.result()
                    pending = None
                    stop = self.output(out, out_now, display)
                    if stop:
                        break
                self._now = now

                if self.mode == "calibrating":
                    self.calibrator.add(res.direction if res else None,
                                        res.head if res else None)
                    if self.calibrator.done(self._now):
                        calib = self.calibrator.finish()
                        if calib is None:
                            log.warning("calibration failed: face not visible enough")
                            self.mode = "ready"
                        else:
                            calib.save(args.calib_file)
                            print(f"calibrated on {calib.samples} frames, "
                                  f"spread {calib.spread_deg:.1f} deg"
                                  + ("  (high - you moved; press C again)"
                                     if calib.spread_deg > 15 else ""))
                            self.apply_calibration(calib)
                            if not self.live and not rewound:
                                # annotate the whole clip, not just what follows calibration
                                self.source.rewind()
                                self.pipe.reset()
                                frame_i, rewound = 0, True
                                continue
                elif self.mode == "monitoring":
                    if self.session_t0 is None:
                        self.session_t0 = self._now
                    state = self.st.update(res.direction if res else None, self._now,
                                           res.head if res else None)
                    self.timeline.add(self._now, self.st.last_angle,
                                      self.st.last_head_angle, state)
                    if state == ALERT:
                        self.beeper.maybe_beep()
                    if self.csv_writer:
                        self.csv_writer.writerow([
                            f"{self._now:.3f}", state,
                            "" if self.st.last_angle is None else f"{self.st.last_angle:.2f}",
                            "" if self.st.last_head_angle is None
                            else f"{self.st.last_head_angle:.2f}", int(res is not None)])

                t = time.perf_counter()
                inst = 1.0 / max(t - prev, 1e-6)
                prev = t
                fps_ema = inst if fps_ema == 0 else 0.92 * fps_ema + 0.08 * inst

                hud = self.hud(fps_ema, now)
                pending = executor.submit(lambda f=frame, r=res, h=hud: (self.render(f, r, h),
                                                                         h.now))
                frame_i += 1
                if args.max_frames and frame_i >= args.max_frames:
                    break
            if pending is not None and not stop:
                out, out_now = pending.result()
                self.output(out, out_now, display)
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown(wait=True)
            self.source.release()
            self.stop_recording()
            if self.csv_file:
                self.csv_file.close()
            if display:
                cv2.destroyAllWindows()

        if self.mode == "monitoring":
            print(f"\nsession: {self.st.alert_count} attention alerts, "
                  f"{self.st.total_away:.1f}s away from the screen")
        return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.no_display:
        enable_dpi_awareness()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    try:
        app = App(args)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
