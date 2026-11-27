"""Measure per-model and end-to-end latency on GPU and CPU.

    python tools/benchmark.py                       # both devices, sample video
    python tools/benchmark.py --device cuda --frames 300

Writes a CSV row per device to results/benchmark.csv and prints a table.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2  # noqa: E402

from gaze_monitor.pipeline import GazePipeline  # noqa: E402


def median_ms(fn, runs):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def bench(device, frames, source, runs):
    pipe = GazePipeline(os.path.join(ROOT, "models"), device=device)
    pipe.warmup(5)
    det = median_ms(lambda: pipe.det.run(np.zeros((1, 1, 480, 640), np.float32)), runs)
    mesh = median_ms(lambda: pipe.mesh_model.run(np.zeros((1, 3, 128, 128), np.float32)), runs)
    gaze2 = median_ms(lambda: pipe.gaze_model.run(np.zeros((2, 96, 160), np.float32)), runs)

    cap = cv2.VideoCapture(source)
    imgs = []
    while len(imgs) < frames:
        ok, f = cap.read()
        if not ok:
            if not imgs:
                raise SystemExit(f"could not read {source}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        imgs.append(f)
    cap.release()
    for f in imgs[:5]:
        pipe(f)
    found = 0
    t0 = time.perf_counter()
    for f in imgs:
        found += pipe(f) is not None
    total = time.perf_counter() - t0
    return {
        "device": pipe.device_label, "face_det_ms": round(det, 2),
        "facemap_ms": round(mesh, 2), "eyegaze_x2_ms": round(gaze2, 2),
        "pipeline_ms": round(total / len(imgs) * 1000, 2),
        "pipeline_fps": round(len(imgs) / total, 1),
        "frames": len(imgs), "face_found": found,
        "resolution": f"{imgs[0].shape[1]}x{imgs[0].shape[0]}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cuda", "cpu", "both"), default="both")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--source", default=os.path.join(ROOT, "samples", "desk-worker.mp4"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "benchmark.csv"))
    args = ap.parse_args()

    devices = ("cuda", "cpu") if args.device == "both" else (args.device,)
    rows = [bench(d, args.frames, args.source, args.runs) for d in devices]

    keys = list(rows[0].keys())
    print("\n" + " | ".join(f"{k:>13}" for k in keys))
    for r in rows:
        print(" | ".join(f"{str(r[k]):>13}" for k in keys))
    if len(rows) == 2:
        print(f"\nGPU speed-up end to end: {rows[1]['pipeline_ms'] / rows[0]['pipeline_ms']:.1f}x")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
