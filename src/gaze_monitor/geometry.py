"""Pre- and post-processing for the three models, and gaze geometry.

Ported from the reference implementation's gaze_common.py (Samuel Alexander,
Apache-2.0) and modified: GPU batching, HWC decode inputs, OpenCV peak suppression.
Every constant here was measured against the models' training pipelines; see the
comments before changing any of them.

    face_det_lite  480x640 grayscale  -> face boxes (+5 points)
    facemap_3dmm   128x128 RGB        -> 265 3DMM params -> 68 landmarks + head pose
    eyegaze        96x160 grayscale   -> per-eye (pitch, yaw) + 34 eye landmarks
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# stage 1: face detection
# ---------------------------------------------------------------------------

FACE_W, FACE_H = 640, 480

# The model zoo preprocesses with skimage.color.rgb2gray, whose luminance weights
# differ from OpenCV's (0.299/0.587/0.114). Matching them keeps the input on the
# distribution the detector was trained on.
_RGB2GRAY_SKIMAGE = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)
# the same weights in BGR channel order, with the /255 folded in, for cv2.transform
_BGR2GRAY_SCALED = (_RGB2GRAY_SKIMAGE[::-1] / 255.0).reshape(1, 3).astype(np.float32)


def face_input(bgr: np.ndarray):
    """BGR uint8 frame -> ((480, 640) float32 in [0, 1], letterbox transform).

    The detector's input is 4:3. Squashing a 16:9 webcam frame into that distorts
    faces enough to wreck the boxes, so pad instead of stretch and return the
    (scale, pad_x, pad_y) needed to undo it.
    """
    h, w = bgr.shape[:2]
    scale = min(FACE_W / w, FACE_H / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (FACE_W - nw) // 2, (FACE_H - nh) // 2
    # Same arithmetic as (rgb / 255) @ weights, but vectorised by OpenCV: ~15x faster
    # than numpy's matmul over a trailing axis of 3. Padding stays exactly 0.
    g = cv2.transform(resized.astype(np.float32), _BGR2GRAY_SCALED)
    if g.ndim == 3:
        g = g[..., 0]
    gray = np.zeros((FACE_H, FACE_W), dtype=np.float32)
    gray[pad_y:pad_y + nh, pad_x:pad_x + nw] = g
    return gray, (scale, pad_x, pad_y)


def _peak_mask(hm: np.ndarray) -> np.ndarray:
    """3x3 max-pool peak suppression (the zoo's F.max_pool2d(3, 1, 1))."""
    dil = cv2.dilate(hm, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT,
                     borderValue=-np.inf)
    return hm >= dil


def detect_faces(heatmap, bbox, landmark, transform, frame_w, frame_h,
                 threshold=0.55, stride=8, max_faces=8):
    """Decode face_det_lite outputs (HWC arrays) into boxes in frame coordinates.

    heatmap (H, W) logits, bbox (H, W, 4), landmark (H, W, 10) or None.
    Returns a list of dicts with 'box' (x0, y0, x1, y1), 'score' and 'landmark5',
    best first.
    """
    hm = 1.0 / (1.0 + np.exp(-heatmap.astype(np.float32)))
    peaks = _peak_mask(hm) & (hm >= threshold)
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return []
    order = np.argsort(hm[ys, xs])[::-1][:max_faces]
    scale, pad_x, pad_y = transform

    faces = []
    for i in order:
        cy, cx = int(ys[i]), int(xs[i])
        x, y, r, b = bbox[cy, cx, :]
        x0, y0, x1, y1 = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x0, x1 = (x0 - pad_x) / scale, (x1 - pad_x) / scale
        y0, y1 = (y0 - pad_y) / scale, (y1 - pad_y) / scale
        face = {
            "box": (float(np.clip(x0, 0, frame_w - 1)), float(np.clip(y0, 0, frame_h - 1)),
                    float(np.clip(x1, 0, frame_w - 1)), float(np.clip(y1, 0, frame_h - 1))),
            "score": float(hm[cy, cx]),
            "landmark5": None,
        }
        # 5 points (eyes, nose, mouth corners): 5 x-offsets then 5 y-offsets, in cells
        if landmark is not None:
            off = landmark[cy, cx, :]
            px = (off[:5] + cx) * stride
            py = (off[5:] + cy) * stride
            face["landmark5"] = np.stack([(px - pad_x) / scale, (py - pad_y) / scale],
                                         axis=1).astype(np.float32)
        faces.append(face)
    return faces


# ---------------------------------------------------------------------------
# stage 2: 68 landmarks + head pose (3D morphable model)
# ---------------------------------------------------------------------------

VERTEX_NUM, ALPHA_ID_SIZE, ALPHA_EXP_SIZE = 68, 219, 39
MESH_SIZE = 128


class FaceMesh:
    """Decodes facemap_3dmm's 265 parameters into 2D landmarks and head pose."""

    def __init__(self, basis_dir: str):
        self.mean = np.load(os.path.join(basis_dir, "meanFace.npy")).reshape(3 * VERTEX_NUM, 1)
        self.basis_id = np.load(os.path.join(basis_dir, "shapeBasis.npy")).reshape(
            3 * VERTEX_NUM, ALPHA_ID_SIZE)
        self.basis_exp = np.load(os.path.join(basis_dir, "blendShape.npy")).reshape(
            3 * VERTEX_NUM, ALPHA_EXP_SIZE)

    @staticmethod
    def crop_input(bgr: np.ndarray, box, size: int = MESH_SIZE):
        """Face box -> ((128, 128, 3) float32 RGB in [0, 1], integer box used)."""
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w - 1), min(y1, h - 1)
        crop = bgr[y0:y1 + 1, x0:x1 + 1]
        if crop.size == 0 or x1 <= x0 or y1 <= y0:
            return None, None
        resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = resized[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)
        return rgb, (x0, y0, x1, y1)

    def decode(self, params, box, size: int = MESH_SIZE):
        """265-vector -> (68x2 landmarks in frame coords, (pitch, yaw, roll) radians, geom)."""
        p = np.asarray(params, dtype=np.float32).reshape(-1)
        alpha_id = p[0:219] * 3.0
        alpha_exp = p[219:258] * 0.5 + 0.5
        pitch, yaw, roll = p[258] * np.pi / 2, p[259] * np.pi / 2, p[260] * np.pi / 2
        tX, tY, tZ = p[261] * 60.0, p[262] * 60.0, 500.0
        f = p[263] * 150.0 + 450.0

        r_matrix = head_rotation((pitch, yaw, roll)).astype(np.float32)
        shape = (self.mean + self.basis_id @ alpha_id.reshape(ALPHA_ID_SIZE, 1)
                 + self.basis_exp @ alpha_exp.reshape(ALPHA_EXP_SIZE, 1)
                 ).reshape(VERTEX_NUM, 3)
        verts = shape @ r_matrix.T
        verts[:, 0] += tX
        verts[:, 1] += tY
        verts[:, 2] += tZ
        lmk = verts[:, 0:2] * f / tZ

        x0, y0, x1, y1 = box
        sx, sy = (x1 - x0 + 1) / size, (y1 - y0 + 1) / size
        lmk[:, 0] = (lmk[:, 0] + size / 2.0) * sx + x0
        lmk[:, 1] = (lmk[:, 1] + size / 2.0) * sy + y0
        geom = {"verts": verts, "focal": float(f), "tZ": float(tZ),
                "to_frame": (sx, sy, float(x0), float(y0), size / 2.0)}
        return lmk, np.array([pitch, yaw, roll], dtype=np.float32), geom


# iBUG-68 indices. 36-41 is the subject's RIGHT eye (image left).
RIGHT_EYE_IDX = list(range(36, 42))
LEFT_EYE_IDX = list(range(42, 48))
# The patch geometry is defined by the two eye corners alone, not the six-point contour.
RIGHT_EYE_CORNERS = (36, 39)
LEFT_EYE_CORNERS = (42, 45)


# ---------------------------------------------------------------------------
# stage 3: eye patches and gaze
# ---------------------------------------------------------------------------

EYE_W, EYE_H = 160, 96
EYE_WIDTH_FACTOR = 1.5


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def eye_patch(gray, landmarks, corners, flip=False, ow=EYE_W, oh=EYE_H):
    """Warp an eye into the model's canonical 160x96 frame.

    The eye centre is the midpoint of the two corners, lands exactly at the patch
    centre, and the patch spans 1.5x the corner-to-corner distance: the geometry the
    model was trained on. A plain rectangular crop is NOT equivalent.

    The model was trained on left eyes, so the right eye is mirrored (flip=True) and
    its yaw negated afterwards.

    Returns (patch_uint8, centre_xy, inverse_transform) or (None, None, None).
    """
    p1 = np.asarray(landmarks[corners[0]], dtype=np.float64)
    p2 = np.asarray(landmarks[corners[1]], dtype=np.float64)
    eye_width = EYE_WIDTH_FACTOR * float(np.linalg.norm(p1 - p2))
    if eye_width <= 1e-3:
        return None, None, None
    cx, cy = 0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])
    s = ow / eye_width
    transform = np.array([[s, 0, 0.5 * ow - s * cx],
                          [0, s, 0.5 * oh - s * cy],
                          [0, 0, 1.0]])
    patch = cv2.equalizeHist(cv2.warpAffine(gray, transform[:2], (ow, oh)))
    if flip:
        patch = np.ascontiguousarray(np.fliplr(patch))
    return patch, (float(cx), float(cy)), np.linalg.inv(transform)


def gaze_from_output(pitchyaw, flipped: bool) -> np.ndarray:
    """Model output -> (pitch, yaw) radians, un-mirrored if the patch was flipped."""
    g = np.asarray(pitchyaw, dtype=np.float32).reshape(-1)[:2].copy()
    if flipped:
        g[1] = -g[1]
    return g


# EyeNet's 34 landmarks: 0-15 eyelid, 16-31 iris boundary, 32 iris centre, 33 eyeball
IRIS_BOUNDARY = slice(16, 32)
IRIS_CENTRE = 32
HEATMAP_H, HEATMAP_W = 48, 80


def eye_landmarks_to_frame(landmarks, inv_transform, flipped: bool):
    """EyeGaze `landmarks` ((y, x) in 48x80 heatmap space) -> (34, 2) frame points."""
    lm = np.asarray(landmarks, dtype=np.float64).reshape(-1, 2)
    if lm.shape[0] < 34:
        return None
    lm = lm * np.array([EYE_H / HEATMAP_H, EYE_W / HEATMAP_W])
    pts = np.empty((lm.shape[0], 3), dtype=np.float64)
    pts[:, 0] = (EYE_W - lm[:, 1]) if flipped else lm[:, 1]
    pts[:, 1] = lm[:, 0]
    pts[:, 2] = 1.0
    return (pts @ inv_transform.T)[:, :2]


def iris_circle(frame_landmarks):
    if frame_landmarks is None:
        return None, None
    ring = frame_landmarks[IRIS_BOUNDARY]
    centre = frame_landmarks[IRIS_CENTRE]
    return centre, float(np.median(np.linalg.norm(ring - centre, axis=1)))


# ---------------------------------------------------------------------------
# directions
# ---------------------------------------------------------------------------


def pitchyaw_to_vector(pitch: float, yaw: float) -> np.ndarray:
    """(pitch, yaw) radians -> unit vector; camera space x right, y down, z forward."""
    return np.array([-np.cos(pitch) * np.sin(yaw), np.sin(pitch),
                     -np.cos(pitch) * np.cos(yaw)], dtype=np.float32)


def gaze_vector(eye_gaze) -> np.ndarray:
    """EyeGaze output is already camera-relative: do NOT rotate it by head pose.

    The eye patch shows the eye as the camera sees it, head turn included. The
    reference implementation measured this on MPIIGaze ground truth: composing with
    a perfect head pose made the error worse (15.5 -> 23.4 degrees).
    """
    v = pitchyaw_to_vector(float(eye_gaze[0]), float(eye_gaze[1]))
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def head_rotation(head_pose) -> np.ndarray:
    """Head rotation in camera space, built exactly as the 3DMM decode places landmarks."""
    pitch, yaw, roll = (float(v) for v in head_pose)
    flip = np.diag([1.0, -1.0, -1.0])
    cr, sr = np.cos(-roll), np.sin(-roll)
    roll_m = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    cy_, sy_ = np.cos(-yaw), np.sin(-yaw)
    yaw_m = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    cp, sp = np.cos(-pitch), np.sin(-pitch)
    pitch_m = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return yaw_m @ (pitch_m @ (flip @ roll_m))


def head_forward(head_pose) -> np.ndarray:
    """Unit vector the face points along. Facing the lens is (0, 0, -1)."""
    return (head_rotation(head_pose) @ np.array([0.0, 0.0, 1.0])).astype(np.float32)


def angle_between(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def rotation_onto(a, b) -> np.ndarray:
    """Rotation matrix taking unit vector a onto unit vector b (Rodrigues)."""
    a = np.asarray(a, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64)
    b = b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1.0 + c)
