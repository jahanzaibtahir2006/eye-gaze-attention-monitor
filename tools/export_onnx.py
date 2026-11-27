"""Export the three Qualcomm AI Hub models from PyTorch to ONNX for GPU inference.

    python tools/export_onnx.py

Needs the export extras (torch, onnx, qai-hub-models). Each model is checked against
its PyTorch original on random input before it is written. The runtime contract is
documented in src/gaze_monitor/pipeline.py.
"""

from __future__ import annotations

import argparse
import os
import shutil

import torch  # import before onnxruntime: both ship cuDNN DLLs on Windows
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPSET = 17


class EyeGazeNoHeatmaps(nn.Module):
    """EyeGaze without its 1.57 MB heatmap output, which the app never reads."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        _, landmarks, gaze = self.model(image)
        return gaze, landmarks


def load_models():
    from qai_hub_models.utils.asset_loaders import always_answer_prompts

    from qai_hub_models.models.face_det_lite.model import FaceDetLite
    from qai_hub_models.models.facemap_3dmm.model import FaceMap_3DMM
    with always_answer_prompts(True):   # EyeGaze clones its source repo on first load
        from qai_hub_models.models.eyegaze.model import EyeGaze
        eyegaze = EyeGaze.from_pretrained()
    return {
        "face_det_lite": (FaceDetLite.from_pretrained().eval(),
                          torch.rand(1, 1, 480, 640), ["image"],
                          ["heatmap", "bbox", "landmark"], None),
        "facemap_3dmm": (FaceMap_3DMM.from_pretrained().eval(),
                         torch.rand(1, 3, 128, 128), ["image"], ["parameters_3dmm"], None),
        # Static batch of 2 (both eyes): with no dynamic dimension, constant folding
        # removes every shape computation, so the whole graph stays on the GPU and
        # can be captured as a single CUDA graph.
        "eyegaze": (EyeGazeNoHeatmaps(eyegaze).eval(),
                    torch.rand(2, 96, 160), ["image"], ["gaze_pitchyaw", "landmarks"], None),
    }


def export(name, model, sample, in_names, out_names, dynamic, out_dir):
    path = os.path.join(out_dir, f"{name}.onnx")
    tmp = path + ".tmp"
    kwargs = dict(input_names=in_names, output_names=out_names, opset_version=OPSET,
                  dynamic_axes=dynamic, do_constant_folding=True)
    with torch.no_grad():
        try:
            torch.onnx.export(model, (sample,), tmp, dynamo=False, **kwargs)
        except TypeError:  # older torch without the dynamo flag
            torch.onnx.export(model, (sample,), tmp, **kwargs)

    import onnx
    m = onnx.load(tmp)
    onnx.checker.check_model(m)
    onnx.save(m, tmp)  # inline any external data into one file

    # verify against PyTorch on fresh random input (batch 1 for eyegaze too)
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    for batch in ([sample.shape[0], 1] if dynamic else [sample.shape[0]]):
        x = torch.rand(batch, *sample.shape[1:])
        with torch.no_grad():
            ref = model(x)
        ref = ref if isinstance(ref, (tuple, list)) else (ref,)
        got = sess.run(None, {in_names[0]: x.numpy()})
        for o, r, g in zip(out_names, ref, got):
            err = float(np.max(np.abs(r.numpy() - g)))
            scale = float(np.max(np.abs(r.numpy()))) + 1e-6
            print(f"  {name}.{o} batch {batch}: shape {tuple(g.shape)}  "
                  f"max abs err {err:.2e} (rel {err / scale:.1e})")
            if err / scale > 1e-3:
                raise SystemExit(f"{name}.{o}: ONNX does not match PyTorch")
    del sess
    shutil.move(tmp, path)
    print(f"  wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "models"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, (model, sample, ins, outs, dyn) in load_models().items():
        print(f"exporting {name}")
        export(name, model, sample, ins, outs, dyn, args.out)


if __name__ == "__main__":
    main()
