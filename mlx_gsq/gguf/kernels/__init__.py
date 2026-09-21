"""MLX-GSQ-RCO custom quantized kernels.

This package contains fused dequantization + matrix multiplication kernels
for GSQ-RCO quantized weights.

- iq3_s_qmv: IQ3_S QMV (quantized matrix-vector) for decode
- iq3_s_qmm: IQ3_S GMM (quantized matrix-matrix) for prefill
- iq2_s_qmv: IQ2_S QMV for mixed-precision tensors
- q4_k_qmv: Q4_K QMV for fallback tensors
"""

from .iq3_s_qmv import IQ3_S_QMV, create_iq3_s_qmv
from .iq3_s_qmm import IQ3_S_QMM, create_iq3_s_qmm
from .iq2_s_qmv import IQ2_S_QMV, create_iq2_s_qmv
from .q4_k_qmv import Q4_K_QMV, create_q4_k_qmv

__all__ = [
    "IQ3_S_QMV", "IQ3_S_QMM", "IQ2_S_QMV", "Q4_K_QMV",
    "create_iq3_s_qmv", "create_iq3_s_qmm", "create_iq2_s_qmv", "create_q4_k_qmv",
]
