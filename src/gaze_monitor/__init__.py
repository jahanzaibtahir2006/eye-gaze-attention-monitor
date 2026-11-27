"""Eye Gaze Attention Monitor.

A webcam attention monitor built from three chained neural networks
(face detection -> 68 facial landmarks + head pose -> per-eye gaze), running on an
NVIDIA GPU through ONNX Runtime.

Pipeline design and reference implementation by Samuel Alexander
(https://github.com/SamuelAlexander/dragonwing-gaze-attention, Apache-2.0), ported here
from the Qualcomm NPU to desktop GPUs.
"""

__version__ = "1.0.0"
