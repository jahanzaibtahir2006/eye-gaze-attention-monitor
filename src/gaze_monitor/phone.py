"""Android phone camera as a live source, over USB, with no extra window.

Uses the scrcpy server (https://github.com/Genymobile/scrcpy) that ships with the
scrcpy release, but not the scrcpy client: this module starts the server on the phone
through adb, opens its video socket and decodes the raw H.264 stream itself (PyAV).
Nothing is shown except the monitor's own window, so a screen recording contains only
that window.

Requirements: the phone in USB debugging mode and authorised, Android 12+ (camera
capture), and a scrcpy release folder (for adb.exe and scrcpy-server).
"""

from __future__ import annotations

import glob
import logging
import os
import random
import re
import shutil
import socket
import subprocess
import threading
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEVICE_SERVER_PATH = "/data/local/tmp/scrcpy-server.jar"
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def find_scrcpy_dir(hint: str | None = None) -> str:
    """The folder holding scrcpy-server and adb: config hint, PATH, then Downloads."""
    candidates = []
    if hint:
        candidates.append(hint)
    exe = shutil.which("scrcpy")
    if exe:
        candidates.append(os.path.dirname(exe))
    home = os.path.expanduser("~")
    candidates += sorted(glob.glob(os.path.join(home, "Downloads", "scrcpy-win64-v*", "**",
                                                "scrcpy-server"), recursive=True), reverse=True)
    candidates += glob.glob(r"C:\scrcpy*")
    for c in candidates:
        d = c if os.path.isdir(c) else os.path.dirname(c)
        if os.path.exists(os.path.join(d, "scrcpy-server")):
            return d
    raise RuntimeError("scrcpy not found. Download it from "
                       "https://github.com/Genymobile/scrcpy/releases and set "
                       "scrcpy-dir in config.toml")


def scrcpy_version(scrcpy_dir: str) -> str:
    """The server only accepts a client of exactly its own version."""
    exe = os.path.join(scrcpy_dir, "scrcpy.exe" if os.name == "nt" else "scrcpy")
    if os.path.exists(exe):
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10)
        m = re.search(r"scrcpy (\d+\.\d+(?:\.\d+)?)", out.stdout + out.stderr)
        if m:
            return m.group(1)
    m = re.search(r"v(\d+\.\d+(?:\.\d+)?)", scrcpy_dir)
    if m:
        return m.group(1)
    raise RuntimeError(f"cannot tell the scrcpy version in {scrcpy_dir}")


class PhoneCameraSource:
    """Newest-frame webcam-like source backed by an Android phone camera."""

    is_live = True

    def __init__(self, scrcpy_dir: str | None = None, facing: str = "front",
                 max_size: int = 1920, fps: int = 30, aspect: str = "16:9",
                 rotate: int = 0, serial: str | None = None, port: int = 27183,
                 connect_timeout: float = 10.0):
        import av  # noqa: F401 - fail early with a clear error if PyAV is missing

        self.dir = find_scrcpy_dir(scrcpy_dir)
        self.adb = os.path.join(self.dir, "adb.exe" if os.name == "nt" else "adb")
        if not os.path.exists(self.adb):
            self.adb = shutil.which("adb") or self.adb
        self.version = scrcpy_version(self.dir)
        self.serial = serial or self._pick_device()
        self.rotate = ROTATIONS.get(int(rotate) % 360)
        self.fps = float(fps)
        self.port = port
        self.scid = f"{random.randint(0, 0x7FFFFFFF):08x}"
        self._proc = None
        self._sock = None
        self._frame = None
        self._seq = 0
        self._last_read = 0
        self._cond = threading.Condition()
        self._running = True
        self._error: str | None = None

        self._adb("push", os.path.join(self.dir, "scrcpy-server"), DEVICE_SERVER_PATH)
        self._adb("forward", f"tcp:{port}", f"localabstract:scrcpy_{self.scid}")
        args = [
            f"scid={self.scid}", "log_level=info", "tunnel_forward=true",
            "video=true", "audio=false", "control=false", "cleanup=true",
            "video_codec=h264", "video_source=camera", f"camera_facing={facing}",
            f"camera_ar={aspect}", f"max_size={max_size}", f"camera_fps={fps}",
            # we decode the bare stream: no device name, codec or frame headers,
            # but keep the dummy byte that proves the server end is really there
            "send_device_meta=false", "send_codec_meta=false", "send_frame_meta=false",
            "send_dummy_byte=true",
        ]
        cmd = [self.adb, "-s", self.serial, "shell",
               f"CLASSPATH={DEVICE_SERVER_PATH}", "app_process", "/",
               "com.genymobile.scrcpy.Server", self.version, *args]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, errors="replace")
        threading.Thread(target=self._server_log, daemon=True).start()
        self._sock = self._connect(connect_timeout)

        self._thread = threading.Thread(target=self._decode, daemon=True)
        self._thread.start()
        ok, frame = self.read(timeout=connect_timeout)
        if not ok:
            self.release()
            raise RuntimeError(self._error or "phone camera sent no frames")
        self.size = (frame.shape[1], frame.shape[0])
        log.info("phone camera (%s, %s) streaming at %dx%d", self.serial, facing, *self.size)

    # ------------------------------------------------------------------ adb

    def _adb(self, *args, check=True, serial=True):
        cmd = [self.adb] + (["-s", self.serial] if serial and getattr(self, "serial", None)
                            else []) + list(args)
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if check and out.returncode != 0:
            raise RuntimeError(f"adb {' '.join(args)} failed: {out.stderr.strip()}")
        return out.stdout

    def _pick_device(self) -> str:
        out = self._adb("devices", serial=False)
        devices = [ln.split("\t") for ln in out.splitlines()[1:] if "\t" in ln]
        ready = [d for d, state in devices if state.strip() == "device"]
        if ready:
            return ready[0]
        if any(state.strip() == "unauthorized" for _, state in devices):
            raise RuntimeError("phone is connected but not authorised: tap 'Allow USB "
                               "debugging' on the phone")
        raise RuntimeError("no phone found: connect it by USB with USB debugging on")

    def _server_log(self):
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            if "ERROR" in line or "Exception" in line:
                self._error = line
                log.error("phone: %s", line)
            else:
                log.debug("phone: %s", line)

    def _connect(self, timeout: float) -> socket.socket:
        """With a forward tunnel adb accepts at once even before the server listens,
        then closes; the server's dummy byte is what proves the link is live."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(self._error or "scrcpy server exited on the phone")
            s = socket.create_connection(("127.0.0.1", self.port), timeout=2.0)
            try:
                if s.recv(1):
                    s.settimeout(5.0)
                    return s
            except OSError:
                pass
            s.close()
            time.sleep(0.15)
        raise RuntimeError(self._error or "could not connect to the phone camera")

    # ------------------------------------------------------------------ video

    def _decode(self):
        import av
        codec = av.CodecContext.create("h264", "r")
        try:
            codec.flags |= av.codec.context.Flags.low_delay
        except AttributeError:
            pass
        while self._running:
            try:
                data = self._sock.recv(1 << 16)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                self._error = self._error or "phone stream ended"
                break
            try:
                for packet in codec.parse(data):
                    for frame in codec.decode(packet):
                        img = frame.to_ndarray(format="bgr24")
                        if self.rotate is not None:
                            img = cv2.rotate(img, self.rotate)
                        with self._cond:
                            self._frame = img
                            self._seq += 1
                            self._cond.notify_all()
            except av.error.FFmpegError as exc:  # a corrupt packet: skip, keep going
                log.debug("decode: %s", exc)
        with self._cond:
            self._running = False
            self._cond.notify_all()

    def cycle_rotation(self) -> int:
        """Rotate the picture another 90 degrees (key O): however the phone is mounted."""
        order = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        self.rotate = order[(order.index(self.rotate) + 1) % 4]
        return {None: 0, cv2.ROTATE_90_CLOCKWISE: 90, cv2.ROTATE_180: 180,
                cv2.ROTATE_90_COUNTERCLOCKWISE: 270}[self.rotate]

    def read(self, timeout: float = 2.0) -> tuple[bool, np.ndarray | None]:
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._seq > self._last_read or not self._running, timeout)
            if not ok or self._seq <= self._last_read:
                return False, None
            self._last_read = self._seq
            return True, self._frame

    def release(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        try:
            self._adb("forward", "--remove", f"tcp:{self.port}", check=False)
        except (OSError, subprocess.SubprocessError):
            pass
