"""Trusted CPU reference path backed by llama.cpp's gguf-py implementation.

This module is a correctness oracle, not the production inference path.
"""
from __future__ import annotations

import numpy as np


def dequantize_reference(packed: bytes | np.ndarray, qtype: str) -> np.ndarray:
    from gguf import GGMLQuantizationType, GGML_QUANT_SIZES, dequantize

    quant_type = GGMLQuantizationType[qtype]
    _, block_bytes = GGML_QUANT_SIZES[quant_type]
    raw = np.frombuffer(packed, dtype=np.uint8) if isinstance(packed, bytes) else np.asarray(packed, dtype=np.uint8)
    if raw.size % block_bytes:
        raise ValueError(f"{qtype} payload size {raw.size} is not divisible by {block_bytes}")
    return dequantize(raw.reshape(-1, block_bytes), quant_type).reshape(-1)


def iq3_s_grid() -> np.ndarray:
    """Return the canonical 512 x 4 IQ3_S codebook."""
    from gguf.quants import IQ3_S

    IQ3_S.init_grid()
    assert IQ3_S.grid is not None
    return np.asarray(IQ3_S.grid, dtype=np.float32).reshape(512, 4)
