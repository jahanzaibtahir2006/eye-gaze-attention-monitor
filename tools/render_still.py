"""Render one image through the pipeline in every view (for docs and quick checks).

    python tools/render_still.py samples/portrait-6471976.jpg --out docs
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2  # noqa: E402

from gaze_monitor.attention import AttentionState, Calibration  # noqa: E402
from gaze_monitor.overlay import (HudInfo, StageRenderer, Timeline,  # noqa: E402
                                  compose_pipeline, compose_studio, draw_monitor)
from gaze_monitor.pipeline import GazePipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    pipe = GazePipeline(os.path.join(ROOT, "models"), device=args.device)
    pipe.warmup()
    res = pipe(img)
    if res is None:
        raise SystemExit("no face found")
    print(f"box {[round(v) for v in res.box]} score {res.score:.2f}")
    print(f"head yaw/pitch/roll deg {[round(float(v), 1) for v in __import__('numpy').degrees(res.pose)]}")
    print(f"gaze direction {res.direction.round(3)}  timings {pipe.timings}")

    st = AttentionState(Calibration.default())
    st.update(res.direction, 0.0, res.head)
    hud = HudInfo(fps=30.0, device="GPU", gpu_name="GPU", timings=pipe.timings,
                  mode="monitoring")
    os.makedirs(args.out, exist_ok=True)
    stages = StageRenderer()
    for name, img_out in (
            ("studio", compose_studio(img, res, st, hud, stages, Timeline(), "Eye Gaze Attention Monitor")),
            ("pipeline", compose_pipeline(img, res, st, hud, stages, "Eye Gaze Attention Monitor")),
            ("monitor", draw_monitor(img, res, st, hud))):
        p = os.path.join(args.out, f"still-{name}.png")
        cv2.imwrite(p, img_out)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
