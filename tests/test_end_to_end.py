"""End-to-end: the real ONNX models on the sample portrait and video."""

import os

import cv2
import numpy as np
import pytest

from gaze_monitor.attention import ALERT, AttentionState, Calibration
from gaze_monitor.overlay import (HudInfo, StageRenderer, Timeline, compose_pipeline,
                                  compose_studio, draw_monitor)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
SAMPLES = os.path.join(ROOT, "samples")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODELS, "eyegaze.onnx")), reason="models not exported")


@pytest.fixture(scope="module", params=["cuda", "cpu"])
def pipe(request):
    from gaze_monitor.pipeline import GazePipeline
    return GazePipeline(MODELS, device=request.param)


def test_portrait_chain(pipe):
    img = cv2.imread(os.path.join(SAMPLES, "portrait-6471976.jpg"))
    res = pipe(img)
    assert res is not None, "no face found"
    x0, y0, x1, y1 = res.box
    assert x1 - x0 > 50 and y1 - y0 > 50
    assert res.landmarks.shape == (68, 2)
    # landmarks fall inside (a slightly grown) face box
    assert np.all(res.landmarks[:, 0] > x0 - 0.2 * (x1 - x0))
    assert np.all(res.landmarks[:, 0] < x1 + 0.2 * (x1 - x0))
    assert len(res.eyes) == 2
    # the subject's right eye is on the image left
    right = next(e for e in res.eyes if e.side == "right")
    left = next(e for e in res.eyes if e.side == "left")
    assert right.iris_centre[0] < left.iris_centre[0]
    assert np.linalg.norm(res.direction) == pytest.approx(1.0, abs=1e-5)
    # both eyes broadly agree (model error is ~15 deg each)
    cos = float(right.gaze @ left.gaze)
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))) < 45


def test_cpu_and_gpu_agree():
    from gaze_monitor.pipeline import GazePipeline
    img = cv2.imread(os.path.join(SAMPLES, "portrait-6471976.jpg"))
    a = GazePipeline(MODELS, device="cuda")(img)
    b = GazePipeline(MODELS, device="cpu")(img)
    assert np.allclose(a.box, b.box, atol=1.0)
    assert np.degrees(np.arccos(np.clip(a.direction @ b.direction, -1, 1))) < 2.0


def test_renderers_produce_1080p(pipe):
    img = cv2.imread(os.path.join(SAMPLES, "portrait-6471976.jpg"))
    res = pipe(img)
    st = AttentionState(Calibration.default())
    hud = HudInfo(fps=30, device="GPU", gpu_name="GPU", timings=pipe.timings,
                  mode="monitoring")
    stages = StageRenderer()
    assert compose_studio(img, res, st, hud, stages, Timeline(), "t").shape == (1080, 1920, 3)
    assert compose_pipeline(img, res, st, hud, stages, "t").shape == (1080, 1920, 3)
    assert draw_monitor(img, res, st, hud).shape == img.shape
    # and with no face at all
    assert compose_studio(img * 0, None, st, hud, StageRenderer(), Timeline(),
                          "t").shape == (1080, 1920, 3)


def test_sleeping_worker_raises_alert(pipe):
    """The desk-worker clip shows someone asleep: the monitor must alert."""
    cap = cv2.VideoCapture(os.path.join(SAMPLES, "desk-worker.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    st = AttentionState(Calibration.default())
    states, i = [], 0
    pipe.reset()
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = pipe(f)
        states.append(st.update(r.direction if r else None, i / fps, r.head if r else None))
        i += 1
    assert i > 100
    assert ALERT in states
