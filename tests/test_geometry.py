import numpy as np
import pytest

from gaze_monitor.geometry import (
    EYE_H,
    EYE_W,
    angle_between,
    detect_faces,
    eye_patch,
    face_input,
    gaze_from_output,
    gaze_vector,
    head_forward,
    rotation_onto,
)


def test_letterbox_keeps_aspect_and_inverts():
    frame = np.full((720, 1280, 3), 200, np.uint8)
    gray, (scale, px, py) = face_input(frame)
    assert gray.shape == (480, 640)
    assert scale == pytest.approx(0.5)
    assert px == 0 and py == 60
    assert gray[0, 0] == 0.0                    # padding is black
    assert gray[240, 320] == pytest.approx(200 / 255, abs=1e-3)


def test_face_input_matches_skimage_grayscale():
    """The fast OpenCV path must equal the zoo's (rgb / 255) @ skimage weights."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    gray, _ = face_input(frame)
    ref = (frame[:, :, ::-1].astype(np.float32) / 255.0) @ np.array(
        [0.2125, 0.7154, 0.0721], np.float32)
    assert np.max(np.abs(gray - ref)) < 1e-5


def test_detect_decodes_single_peak_back_to_frame():
    hm = np.full((60, 80), -10.0, np.float32)
    hm[30, 40] = 5.0
    bbox = np.zeros((60, 80, 4), np.float32)
    bbox[30, 40] = [2, 3, 2, 3]                  # cells left, up, right, down
    faces = detect_faces(hm, bbox, None, (0.5, 0, 60), 1280, 720)
    assert len(faces) == 1
    x0, y0, x1, y1 = faces[0]["box"]
    # centre cell (40, 30) * stride 8 = (320, 240) in letterbox -> (640, 360) in frame
    assert (x0 + x1) / 2 == pytest.approx(640)
    assert (y0 + y1) / 2 == pytest.approx(360)
    assert x1 - x0 == pytest.approx(4 * 8 / 0.5)


def test_eye_patch_centres_eye_and_spans_1_5x():
    gray = np.zeros((400, 600), np.uint8)
    lmk = np.zeros((68, 2))
    lmk[36] = (200, 150)
    lmk[39] = (260, 150)
    patch, centre, inv = eye_patch(gray, lmk, (36, 39))
    assert patch.shape == (EYE_H, EYE_W)
    assert centre == (230.0, 150.0)
    # patch centre maps back to the eye centre; patch width covers 1.5 x 60 px
    c = inv @ np.array([EYE_W / 2, EYE_H / 2, 1.0])
    assert c[:2] == pytest.approx([230, 150])
    left = inv @ np.array([0, EYE_H / 2, 1.0])
    right = inv @ np.array([EYE_W, EYE_H / 2, 1.0])
    assert right[0] - left[0] == pytest.approx(90)


def test_right_eye_yaw_is_negated():
    g = gaze_from_output(np.array([0.1, 0.3]), flipped=True)
    assert g[0] == pytest.approx(0.1) and g[1] == pytest.approx(-0.3)


def test_straight_gaze_points_at_camera():
    v = gaze_vector([0.0, 0.0])
    assert v == pytest.approx([0, 0, -1])


def test_frontal_head_faces_camera():
    assert head_forward([0.0, 0.0, 0.0]) == pytest.approx([0, 0, -1], abs=1e-6)


def test_angle_between():
    assert angle_between([1, 0, 0], [0, 1, 0]) == pytest.approx(90)


def test_rotation_onto():
    a = gaze_vector([0.3, -0.2])
    R = rotation_onto(a, [0, 0, -1])
    assert R @ a == pytest.approx([0, 0, -1], abs=1e-6)
