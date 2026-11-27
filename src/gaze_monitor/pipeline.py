"""The three-model chain: face -> landmarks + head pose -> both eyes -> one gaze direction.

ONNX model contract (produced by tools/export_onnx.py):

    face_det_lite.onnx  image (1, 1, 480, 640)  -> heatmap (1, 1, 60, 80),
                                                   bbox (1, 4, 60, 80),
                                                   landmark (1, 10, 60, 80)
    facemap_3dmm.onnx   image (1, 3, 128, 128)  -> parameters_3dmm (1, 265)
    eyegaze.onnx        image (2, 96, 160)      -> gaze_pitchyaw (2, 2),
                                                   landmarks (2, 34, 2)

EyeGaze is exported with a batch of two, so both eyes go through the GPU in a single
call, and without the 1.57 MB heatmap output the application never reads. All shapes
are static so each model can run as one captured CUDA graph (see runtime.py).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from .geometry import (
    EYE_H,
    EYE_W,
    LEFT_EYE_CORNERS,
    LEFT_EYE_IDX,
    RIGHT_EYE_CORNERS,
    RIGHT_EYE_IDX,
    FaceMesh,
    detect_faces,
    eye_landmarks_to_frame,
    eye_patch,
    face_input,
    gaze_from_output,
    gaze_vector,
    head_forward,
    iris_circle,
    rotation_onto,
    to_gray,
)
from .filters import DirectionFilter, OneEuro
from .runtime import OnnxModel


@dataclass
class Eye:
    side: str                    # "right" (subject's right, image left) or "left"
    patch: np.ndarray            # 96x160 uint8, exactly what the model saw
    quad: np.ndarray             # (4, 2) patch corners in frame coordinates
    iris_centre: np.ndarray
    iris_radius: float
    gaze: np.ndarray             # unit vector, camera space (raw model output)
    display_gaze: np.ndarray     # One-Euro smoothed, for drawing only
    landmark_idx: list[int]


@dataclass
class FrameResult:
    box: tuple[float, float, float, float]
    score: float
    landmark5: np.ndarray | None
    landmarks: np.ndarray        # (68, 2)
    pose: np.ndarray             # pitch, yaw, roll (radians)
    head: np.ndarray             # unit vector the face points along
    direction: np.ndarray        # mean gaze of both eyes, unit vector (raw)
    display_direction: np.ndarray  # One-Euro smoothed, for drawing only
    eyes: list[Eye] = field(default_factory=list)
    arrow_delta: np.ndarray | None = None


class GazePipeline:
    def __init__(self, model_dir: str, device: str = "cuda", detect_every: int = 1,
                 det_threshold: float = 0.55, arrow_scale: float = 1.9):
        self.det = OnnxModel(os.path.join(model_dir, "face_det_lite.onnx"), device)
        self.mesh_model = OnnxModel(os.path.join(model_dir, "facemap_3dmm.onnx"), device)
        self.gaze_model = OnnxModel(os.path.join(model_dir, "eyegaze.onnx"), device)
        self.mesh = FaceMesh(os.path.join(model_dir, "3dmm"))
        self.detect_every = max(1, detect_every)
        self.det_threshold = det_threshold
        self.arrow_scale = arrow_scale
        self.timings: dict[str, float] = {}
        self._frame_i = 0
        self._last_face = None
        self._to_reference = np.eye(3)
        # Drawing filters. The attention state machine keeps using the raw output
        # (it has its own, measured smoothing); these only stop the arrows flickering.
        self._f_dir = DirectionFilter(min_cutoff=0.3, beta=0.5)
        self._f_eye = {s: DirectionFilter(min_cutoff=0.3, beta=0.5) for s in ("right", "left")}
        self._f_iris = {s: OneEuro(min_cutoff=1.5, beta=0.02) for s in ("right", "left")}

    def _reset_filters(self) -> None:
        self._f_dir.reset()
        for f in (*self._f_eye.values(), *self._f_iris.values()):
            f.reset()

    @property
    def device_label(self) -> str:
        models = (self.det, self.mesh_model, self.gaze_model)
        return "GPU (CUDA)" if all(m.on_gpu for m in models) else (
            "CPU" if not any(m.on_gpu for m in models) else "GPU+CPU")

    def warmup(self, runs: int = 3) -> None:
        """First CUDA inferences pick kernels and allocate memory; do it off-camera."""
        for _ in range(runs):
            self.det.run(np.zeros((1, 1, 480, 640), np.float32))
            self.mesh_model.run(np.zeros((1, 3, 128, 128), np.float32))
            self.gaze_model.run(np.zeros((2, EYE_H, EYE_W), np.float32))

    def reset(self) -> None:
        self._frame_i = 0
        self._last_face = None
        self._reset_filters()

    def set_reference(self, axis) -> None:
        """Draw arrows relative to the calibrated axis, so they shrink to a dot on task.

        The camera usually sits above the screen, so a reader looks ~20 degrees below
        the lens and a camera-relative arrow would point down the whole time.
        """
        self._to_reference = rotation_onto(axis, [0.0, 0.0, -1.0])

    # ------------------------------------------------------------------ stages

    def _detect(self, frame):
        h, w = frame.shape[:2]
        gray, transform = face_input(frame)
        out = self.det.run(gray[None, None])
        heatmap = out["heatmap"][0, 0]
        bbox = np.transpose(out["bbox"][0], (1, 2, 0))
        lmk5 = out.get("landmark")
        lmk5 = np.transpose(lmk5[0], (1, 2, 0)) if lmk5 is not None else None
        return detect_faces(heatmap, bbox, lmk5, transform, w, h,
                            threshold=self.det_threshold)

    def __call__(self, frame: np.ndarray, now: float | None = None) -> FrameResult | None:
        now = time.perf_counter() if now is None else now
        t: dict[str, float] = {}

        t0 = time.perf_counter()
        if self._frame_i % self.detect_every == 0 or self._last_face is None:
            faces = self._detect(frame)
            self._last_face = faces[0] if faces else None
        self._frame_i += 1
        t["detect"] = (time.perf_counter() - t0) * 1000
        face = self._last_face
        if face is None:
            self.timings = t
            self._reset_filters()
            return None

        t0 = time.perf_counter()
        crop, box_used = FaceMesh.crop_input(frame, face["box"])
        if crop is None:
            self.timings = t
            return None
        params = self.mesh_model.run(np.transpose(crop, (2, 0, 1))[None])["parameters_3dmm"]
        lmk, pose, _ = self.mesh.decode(params, box_used)
        t["landmarks"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        gray = to_gray(frame)
        prepared = []
        for side, corners, idx, flip in (("right", RIGHT_EYE_CORNERS, RIGHT_EYE_IDX, True),
                                         ("left", LEFT_EYE_CORNERS, LEFT_EYE_IDX, False)):
            patch, centre, inv = eye_patch(gray, lmk, corners, flip=flip)
            if patch is not None:
                prepared.append((side, corners, idx, flip, patch, centre, inv))
        if not prepared:
            self.timings = t
            return None

        # The model has a fixed batch of 2 (for the CUDA graph); pad with a copy when
        # only one eye could be cut out.
        patches = [p[4] for p in prepared]
        if len(patches) == 1:
            patches.append(patches[0])
        batch = np.stack(patches).astype(np.float32) * (1.0 / 255.0)
        out = self.gaze_model.run(batch)

        eyes: list[Eye] = []
        for k, (side, corners, idx, flip, patch, centre, inv) in enumerate(prepared):
            g = gaze_from_output(out["gaze_pitchyaw"][k], flipped=flip)
            frame_lm = eye_landmarks_to_frame(out["landmarks"][k], inv, flip)
            ic, ir = iris_circle(frame_lm)
            span = float(np.linalg.norm(lmk[corners[0]] - lmk[corners[1]]))
            if ic is None:
                ic, ir = np.array(centre), span * 0.24
            else:
                ir = max(float(ir), span * 0.24)
            g_vec = gaze_vector(g)
            ic = self._f_iris[side](ic, now)
            quad = np.array([[0, 0, 1], [EYE_W, 0, 1], [EYE_W, EYE_H, 1], [0, EYE_H, 1]],
                            dtype=np.float64) @ inv.T
            eyes.append(Eye(side=side, patch=patch, quad=quad[:, :2],
                            iris_centre=np.asarray(ic, np.float64), iris_radius=float(ir),
                            gaze=g_vec, display_gaze=self._f_eye[side](g_vec, now),
                            landmark_idx=idx))
        t["gaze"] = (time.perf_counter() - t0) * 1000
        self.timings = t

        # The per-eye errors are largely independent, so the mean beats either eye.
        mean_v = np.mean([e.gaze for e in eyes], axis=0)
        n = np.linalg.norm(mean_v)
        if n < 1e-6:
            return None
        mean_v = mean_v / n
        disp_v = self._f_dir(mean_v, now)

        # One shared 2D arrow for both eyes, length = sin(angle off the calibrated axis)
        if len(eyes) == 2:
            inter = float(np.linalg.norm(eyes[0].iris_centre - eyes[1].iris_centre))
        else:
            bx0, by0, bx1, by1 = face["box"]
            inter = 0.25 * float(np.hypot(bx1 - bx0, by1 - by0))
        rel = self._to_reference @ disp_v.astype(np.float64)
        arrow_delta = rel[:2] * self.arrow_scale * inter if inter > 0 else None

        return FrameResult(box=face["box"], score=face["score"],
                           landmark5=face.get("landmark5"), landmarks=lmk, pose=pose,
                           head=head_forward(pose), direction=mean_v,
                           display_direction=disp_v, eyes=eyes,
                           arrow_delta=arrow_delta)
