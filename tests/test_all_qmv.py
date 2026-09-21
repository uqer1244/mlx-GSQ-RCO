from pathlib import Path
import numpy as np
import pytest

from mlx_gsq.quant import dequantize_reference

QTYPES=("IQ1_M","IQ1_S","IQ2_XXS","IQ2_XS","IQ2_S","IQ3_XXS","IQ3_S","IQ4_XS","Q2_K","Q4_K","Q6_K")

@pytest.mark.parametrize("rows", [2, 5])
@pytest.mark.parametrize("qtype",QTYPES)
def test_fused_qmv_matches_reference(qtype, rows):
    mx=pytest.importorskip("mlx.core")
    if not mx.metal.is_available(): pytest.skip("Metal unavailable")
    from mlx_gsq.runtime import PackedModelStore
    from mlx_gsq.runtime.kernels.qmv import quantized_matvec
    path=Path("Qwen3.8-27B-GSQ-RCO/model-packed.safetensors")
    if not path.exists(): pytest.skip("packed model absent")
    store=PackedModelStore(path)
    name=next(n for n,i in store.manifest["tensors"].items() if i["quantization_type"]==qtype and len(i["logical_shape"])==2)
    w=store.weight(name); K=w.logical_shape[1]; N=4; row_bytes=w.num_bytes//w.logical_shape[0]
    packed=w.data[:N*row_bytes]; rng=np.random.default_rng(123); x_np=rng.normal(size=(rows,K)).astype(np.float32)
    dense=dequantize_reference(np.asarray(packed),qtype).reshape(N,K); expected=x_np@dense.T
    actual=np.asarray(quantized_matvec(mx.array(x_np),packed,qtype=qtype,out_features=N,in_features=K))
    np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=5e-4,err_msg=f"{qtype} {name}")
