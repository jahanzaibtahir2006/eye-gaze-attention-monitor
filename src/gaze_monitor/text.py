"""Anti-aliased TrueType text on OpenCV images (Hershey fonts look dated on video)."""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAVE_PIL = True
except ImportError:  # pragma: no cover
    _HAVE_PIL = False

_FONT_DIRS = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
              "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"]
_FONT_FILES = {
    "regular": ["segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf"],
    "semibold": ["seguisb.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"],
    "bold": ["segoeuib.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"],
    "mono": ["consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"],
}


@lru_cache(maxsize=64)
def _font(weight: str, size: int):
    for name in _FONT_FILES.get(weight, _FONT_FILES["regular"]):
        for d in _FONT_DIRS:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


@lru_cache(maxsize=1024)
def _render(text: str, size: int, weight: str, color: tuple[int, int, int]):
    """Render once to a BGR tile plus alpha mask; cached because labels repeat."""
    font = _font(weight, size)
    left, top, right, bottom = font.getbbox(text)
    w, h = max(1, right - left), max(1, bottom - top)
    img = Image.new("L", (w + 2, h + 2), 0)
    ImageDraw.Draw(img).text((1 - left, 1 - top), text, font=font, fill=255)
    alpha = np.asarray(img, dtype=np.float32)[..., None] / 255.0
    tile = np.empty((alpha.shape[0], alpha.shape[1], 3), np.float32)
    tile[:] = color
    return tile, alpha, (w, h), top


def text_size(text: str, size: int, weight: str = "regular") -> tuple[int, int]:
    if not _HAVE_PIL:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size / 30, 2)
        return w, h
    left, top, right, bottom = _font(weight, size).getbbox(text)
    return right - left, bottom - top


def put_text(img: np.ndarray, text: str, org: tuple[int, int], size: int,
             color=(245, 245, 245), weight: str = "regular", anchor: str = "lt") -> int:
    """Draw text with its top-left (anchor 'lt'), left-middle ('lm') or right-top ('rt')
    at org. Returns the drawn width."""
    if not text:
        return 0
    if not _HAVE_PIL:
        cv2.putText(img, text, (org[0], org[1] + size), cv2.FONT_HERSHEY_SIMPLEX,
                    size / 30, color, 2, cv2.LINE_AA)
        return text_size(text, size, weight)[0]
    tile, alpha, (w, h), _ = _render(text, size, weight, tuple(int(c) for c in color))
    x, y = org
    if anchor[0] == "r":
        x -= w
    elif anchor[0] == "c":
        x -= w // 2
    if anchor[1] == "m":
        y -= tile.shape[0] // 2
    elif anchor[1] == "b":
        y -= tile.shape[0]
    th, tw = tile.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + tw, img.shape[1]), min(y + th, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return w
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - a) + tile[y0 - y:y1 - y, x0 - x:x1 - x] * a).astype(np.uint8)
    return w
