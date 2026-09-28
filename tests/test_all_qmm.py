"""Tiled QMM kernel tests (multi-row matmul for prefill).

QMM decodes each packed weight tile once per tile and reuses it across the
M_TILE=16 activation rows in the tile, which makes it competitive with (and
often faster than) the per-row QMV path for multi-row inputs.

The tiles are stored in half precision, so results are checked with a
half-precision-appropriate tolerance (relative error ~1e-4..1e-3).
"""
from pathlib import Path
import numpy as np
import pytest

from mlx_gsq.quant import dequantize_reference

QTYPES=("IQ1_M","IQ1_S","IQ2_XXS","IQ2_XS","IQ2_S","IQ3_XXS","IQ3_S","IQ4_XS","Q2_K","Q4_K","Q6_K")
BLOCK_BYTES={"IQ1_M":56,"IQ1_S":50,"IQ2_XXS":66,"IQ2_XS":74,"IQ2_S":82,"IQ3_XXS":98,"IQ3_S":110,"IQ4_XS":136,"Q2_K":84,"Q4_K":144,"Q6_K":210}
N=4  # weight rows used per test (a single output column tile)

def _pack(qtype):
    from mlx_gsq.runtime import PackedModelStore
    path=Path("Qwen3.8-27B-GSQ-RCO/model-packed.safetensors")
    if not path.exists(): pytest.skip("packed model absent")
    store=PackedModelStore(path)
    name=next(n for n,i in store.manifest["tensors"].items() if i["quantization_type"]==qtype and len(i["logical_shape"])==2)
    w=store.weight(name)
    packed=w.data[:N*(w.num_bytes//w.logical_shape[0])]
    K=256*(packed.size//(N*BLOCK_BYTES[qtype]))
    return packed,K,name

def _run(qtype,x_np,packed,K,name):
    import mlx.core as mx
    from mlx_gsq.runtime.kernels.qmm import quantized_matmul
    dense=dequantize_reference(np.asarray(packed),qtype).reshape(N,K)
    expected=x_np.astype(np.float32)@dense.T
    actual=np.asarray(quantized_matmul(mx.array(x_np),packed,qtype=qtype,out_features=N,in_features=K))
    # half-precision tiles: tolerance scaled to the output magnitude
    scale=np.abs(expected).max()
    np.testing.assert_allclose(actual,expected,rtol=2e-3,atol=2e-3*scale,err_msg=f"{qtype} {name}")

@pytest.mark.parametrize("rows",[8,16,33])  # 1, 2, 3 row tiles (33 exercises padding)
@pytest.mark.parametrize("qtype",QTYPES)
def test_qmm_matches_reference(qtype,rows):
    mx=pytest.importorskip("mlx.core")
    if not mx.metal.is_available(): pytest.skip("Metal unavailable")
    packed,K,name=_pack(qtype)
    rng=np.random.default_rng(123)
    x_np=rng.normal(size=(rows,K)).astype(np.float32)
    _run(qtype,x_np,packed,K,name)

def test_qmm_fp16_input():
    mx=pytest.importorskip("mlx.core")
    if not mx.metal.is_available(): pytest.skip("Metal unavailable")
    packed,K,name=_pack("Q4_K")
    rng=np.random.default_rng(7)
    x_np=rng.normal(size=(16,K)).astype(np.float16)
    _run("Q4_K",x_np,packed,K,name)
