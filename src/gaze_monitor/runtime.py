"""ONNX Runtime sessions on the GPU (CUDA), with automatic CPU fallback."""

from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

_DLLS_READY = False


def _prepare_cuda_dlls() -> None:
    """Make the CUDA 12 / cuDNN 9 DLLs from the nvidia-* wheels visible to ORT.

    onnxruntime-gpu[cuda,cudnn] installs them as pip packages instead of requiring a
    system CUDA toolkit; preload_dlls() (ORT >= 1.21) loads them from there.
    """
    global _DLLS_READY
    if _DLLS_READY:
        return
    _DLLS_READY = True

    # cuDNN loads some sub-libraries (e.g. cudnn_engines_tensor_ir64_9.dll) lazily by
    # bare name, which only searches PATH. Put every nvidia/*/bin folder on it.
    if os.name == "nt":
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
        bins = []
        for root in roots:
            for pkg in sorted(os.listdir(root)):
                b = os.path.join(root, pkg, "bin")
                if os.path.isdir(b):
                    bins.append(b)
        if bins:
            os.environ["PATH"] = os.pathsep.join(bins + [os.environ.get("PATH", "")])
            for b in bins:
                try:
                    os.add_dll_directory(b)
                except OSError:
                    pass

    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(directory="")
        except Exception as exc:  # noqa: BLE001 - fall back to CPU, never crash here
            log.debug("preload_dlls failed: %s", exc)


def available_providers() -> list[str]:
    import onnxruntime as ort
    return ort.get_available_providers()


class OnnxModel:
    """One ONNX model. `run` takes and returns named numpy arrays.

    On CUDA the whole model is captured once as a CUDA graph and replayed every call.
    These models are dozens of small kernels; on Windows (WDDM) launching them one by
    one leaves the GPU idle most of the time (measured 13% utilisation on a GTX 1050 Ti).
    Replaying a graph is one launch: 2-3x faster here, bit-identical outputs. It needs
    fixed input shapes and fixed device buffers, hence the I/O binding.
    """

    def __init__(self, path: str, device: str = "cuda", device_id: int = 0,
                 cuda_graph: bool = True):
        import onnxruntime as ort

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run `python tools/export_onnx.py` first.")
        self.name = os.path.splitext(os.path.basename(path))[0]
        self._ort = ort
        self._binding = None

        want_cuda = False
        if device == "cuda":
            _prepare_cuda_dlls()
            want_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
            if not want_cuda:
                log.warning("CUDA provider unavailable; %s runs on CPU", self.name)

        self.session = None
        if want_cuda and cuda_graph:
            try:
                self.session = self._make_session(path, device_id, graph=True)
                self._bind(device_id)
                self.run(np.zeros(self._in_shape, np.float32))   # capture the graph now
            except Exception as exc:  # noqa: BLE001 - fall back to plain CUDA
                log.warning("%s: CUDA graph unavailable (%s); plain CUDA", self.name,
                            str(exc).splitlines()[0][:120])
                self.session, self._binding = None, None
        if self.session is None:
            self.session = self._make_session(path, device_id, graph=False,
                                              cuda=want_cuda)

        self.provider = self.session.get_providers()[0]
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.input_shapes = [i.shape for i in self.session.get_inputs()]
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.cuda_graph = self._binding is not None
        log.info("%s on %s%s", self.name, self.provider,
                 " (CUDA graph)" if self.cuda_graph else "")

    def _make_session(self, path, device_id, graph, cuda=True):
        ort = self._ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        providers: list = ["CPUExecutionProvider"]
        if cuda:
            cuda_opts = {
                "device_id": device_id,
                # HEURISTIC measured 2x faster than EXHAUSTIVE on Pascal with cuDNN 9,
                # whose exhaustive search can settle on a slow fallback engine
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            }
            if graph:
                cuda_opts["enable_cuda_graph"] = "1"
                providers = []   # a graph needs every node on the GPU
            providers.insert(0, ("CUDAExecutionProvider", cuda_opts))
        return ort.InferenceSession(path, sess_options=opts, providers=providers)

    def _bind(self, device_id):
        ort = self._ort
        inp = self.session.get_inputs()[0]
        if any(not isinstance(d, int) for d in inp.shape):
            raise ValueError("dynamic input shape")
        self._in_shape = tuple(inp.shape)
        self._in = ort.OrtValue.ortvalue_from_shape_and_type(
            self._in_shape, np.float32, "cuda", device_id)
        self._outs = {}
        b = self.session.io_binding()
        b.bind_ortvalue_input(inp.name, self._in)
        for o in self.session.get_outputs():
            v = ort.OrtValue.ortvalue_from_shape_and_type(tuple(o.shape), np.float32,
                                                          "cuda", device_id)
            b.bind_ortvalue_output(o.name, v)
            self._outs[o.name] = v
        self._binding = b

    @property
    def on_gpu(self) -> bool:
        return self.provider == "CUDAExecutionProvider"

    def run(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x = np.ascontiguousarray(x, np.float32)
        if self._binding is not None:
            self._in.update_inplace(x)
            self.session.run_with_iobinding(self._binding)
            return {k: v.numpy() for k, v in self._outs.items()}
        out = self.session.run(None, {self.session.get_inputs()[0].name: x})
        return dict(zip([o.name for o in self.session.get_outputs()], out))
