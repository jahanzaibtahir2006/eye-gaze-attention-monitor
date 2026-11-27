"""Video input (webcam or file) and H.264 recording."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CameraSource:
    
    is_live = True

    def __init__(self, index: int, width: int = 1280, height: int = 720, fps: int = 30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {index}")
        # MJPG must be requested before the resolution, or many webcams fall back to
        # uncompressed YUY2 and cap out near 10 FPS at 720p.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or float(fps)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        log.info("camera %d opened at %dx%d, %.0f FPS requested", index, *self.size, self.fps)

        self._frame: np.ndarray | None = None
        self._seq = 0
        self._last_read = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._grab, daemon=True)
        self._thread.start()

    def _grab(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 2.0) -> tuple[bool, np.ndarray | None]:
        """Block until a frame newer than the last one returned is available."""
        with self._cond:
            ok = self._cond.wait_for(lambda: self._seq > self._last_read, timeout)
            if not ok:
                return False, None
            self._last_read = self._seq
            return True, self._frame

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


class FileSource:
    """A video file, read frame by frame, with its own media clock."""

    is_live = False

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.cap.read()

    def rewind(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self) -> None:
        self.cap.release()


def open_source(spec: str, width: int, height: int, fps: int, phone: dict | None = None):
    """'phone' = Android phone over USB (falls back to webcam 0), a digit = webcam
    index, anything else = a video file."""
    if spec == "phone":
        from .phone import PhoneCameraSource
        try:
            return PhoneCameraSource(**(phone or {}))
        except Exception as exc:  # noqa: BLE001 - any phone problem: use the webcam
            log.warning("phone camera unavailable (%s); using webcam 0", exc)
            return CameraSource(0, width, height, fps)
    if spec.isdigit():
        return CameraSource(int(spec), width, height, fps)
    return FileSource(spec)


def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


class Recorder:
    """Writes annotated frames to an H.264 MP4 at a constant frame rate.

    Live processing runs at whatever rate the pipeline manages, but a video for social
    media needs a constant rate and real-time speed. Each frame carries a timestamp;
    the recorder repeats or drops frames so that the file plays back at wall-clock
    speed. Uses ffmpeg (libx264, yuv420p) when available, which every player and
    upload site accepts; falls back to OpenCV's mp4v otherwise.
    """

    def __init__(self, path: str, size: tuple[int, int], fps: float = 30.0,
                 crf: int = 18):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path, self.size, self.fps = path, size, float(fps)
        self.frames_written = 0
        self._t0: float | None = None
        self._proc = None
        self._writer = None
        exe = ffmpeg_exe()
        if exe:
            w, h = size
            cmd = [exe, "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                   "-r", f"{self.fps}", "-i", "-",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            log.info("recording H.264 to %s", path)
        else:
            self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                           self.fps, size)
            log.warning("ffmpeg not found; recording with OpenCV mp4v to %s", path)

    def _emit(self, frame: np.ndarray) -> None:
        if self._proc is not None:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self._writer.write(frame)
        self.frames_written += 1

    def write(self, frame: np.ndarray, t: float | None = None) -> None:
        """Write one frame. With `t` (seconds), pace the file to that clock."""
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        if t is None:
            self._emit(frame)
            return
        if self._t0 is None:
            self._t0 = t
        target = int(round((t - self._t0) * self.fps)) + 1
        if self.frames_written == 0:
            self._emit(frame)
        while self.frames_written < target:
            self._emit(frame)

    def close(self) -> None:
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.wait()
        if self._writer is not None:
            self._writer.release()
        log.info("saved %s (%d frames, %.1f s)", self.path, self.frames_written,
                 self.frames_written / self.fps)
