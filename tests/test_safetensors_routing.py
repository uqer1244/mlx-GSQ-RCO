from pathlib import Path
import numpy as np
import pytest

from mlx_gsq.quant import dequantize_reference


def test_all_quant_types_route_from_safetensors_metadata() -> None:
    mx=pytest.importorskip("mlx.core")
    if not mx.metal.is_available(): pytest.skip("Metal unavailable")
    from mlx_gsq.runtime import PackedModelStore
    from mlx_gsq.runtime.kernels.registry import SUPPORTED_QUANT_TYPES,decode_weight
    path=Path("Qwen3.8-27B-GSQ-RCO/model-packed.safetensors")
    if not path.exists(): pytest.skip("packed model absent")
    store=PackedModelStore(path,manifest_path="/tmp/missing-mlx-gsq-manifest.json")
    for qtype in SUPPORTED_QUANT_TYPES:
        name=next(name for name,info in store.manifest["tensors"].items() if info["quantization_type"]==qtype)
        weight=store.weight(name); block_bytes=weight.num_bytes*256//weight.num_elements
        raw=np.asarray(weight.data[:4*block_bytes])
        expected=dequantize_reference(raw,qtype)
        actual=np.asarray(decode_weight(weight,block_count=4))
        np.testing.assert_array_equal(actual,expected,err_msg=f"routing failed for {qtype}: {name}")
