from pathlib import Path
import numpy as np
import pytest

from mlx_gsq.gguf import GGUFReader
from mlx_gsq.quant import dequantize_reference


@pytest.mark.parametrize("qtype", ["IQ1_M","IQ1_S","IQ2_XXS","IQ2_XS","IQ2_S","IQ3_XXS","IQ3_S","IQ4_XS","Q2_K","Q4_K","Q6_K"])
def test_metal_decoder_matches_reference(qtype: str) -> None:
    mx=pytest.importorskip("mlx.core")
    if not mx.metal.is_available(): pytest.skip("Metal unavailable")
    from mlx_gsq.runtime.kernels.registry import decode
    path=Path("Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf")
    if not path.exists(): pytest.skip("local model absent")
    reader=GGUFReader(path); tensor=next(t for t in reader.tensors if t.quantization_type.name==qtype)
    block_bytes={"IQ1_M":56,"IQ1_S":50,"IQ2_XXS":66,"IQ2_XS":74,"IQ2_S":82,"IQ3_XXS":98,"IQ3_S":110,"IQ4_XS":136,"Q2_K":84,"Q4_K":144,"Q6_K":210}[qtype]
    count=64
    with path.open("rb") as stream:
        stream.seek(tensor.absolute_offset); raw=stream.read(count*block_bytes)
    expected=dequantize_reference(raw,qtype)
    actual=np.asarray(decode(mx.array(np.frombuffer(raw,dtype=np.uint8)),qtype))
    np.testing.assert_array_equal(actual,expected)
