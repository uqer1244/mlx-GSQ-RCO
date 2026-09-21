from pathlib import Path

import numpy as np
import pytest

from mlx_gsq.gguf import GGUFReader
from mlx_gsq.quant import dequantize_reference


def test_iq3_s_metal_matches_reference() -> None:
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available(): pytest.skip("Metal unavailable")
    from mlx_gsq.runtime.kernels import decode_iq3_s

    path=Path("Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf")
    if not path.exists(): pytest.skip("local model absent")
    reader=GGUFReader(path); tensor=reader.tensor("blk.0.attn_gate.weight")
    block_count=64; byte_count=block_count*110
    with path.open("rb") as stream:
        stream.seek(tensor.absolute_offset); raw=stream.read(byte_count)
    expected=dequantize_reference(raw,"IQ3_S")
    actual=np.asarray(decode_iq3_s(mx.array(np.frombuffer(raw,dtype=np.uint8))))
    np.testing.assert_array_equal(actual,expected)
