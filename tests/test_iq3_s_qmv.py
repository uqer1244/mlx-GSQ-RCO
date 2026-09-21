from pathlib import Path
import numpy as np
import pytest

from mlx_gsq.quant import dequantize_reference


def test_iq3_s_fused_qmv_matches_reference():
    mx=pytest.importorskip("mlx.core")
    if not mx.metal.is_available(): pytest.skip("Metal unavailable")
    from mlx_gsq.runtime import PackedModelStore
    from mlx_gsq.runtime.kernels.iq3_s_qmv import iq3_s_qmv
    path=Path("Qwen3.8-27B-GSQ-RCO/model-packed.safetensors")
    if not path.exists(): pytest.skip("packed model absent")
    store=PackedModelStore(path); w=store.weight("blk.0.attn_gate.weight")
    K=w.logical_shape[1]; N=8; row_bytes=(K//256)*110
    packed=w.data[:N*row_bytes]
    rng=np.random.default_rng(7); x_np=rng.normal(size=(2,K)).astype(np.float32)
    raw=np.asarray(packed); dense=dequantize_reference(raw,"IQ3_S").reshape(N,K)
    expected=x_np@dense.T
    actual=np.asarray(iq3_s_qmv(mx.array(x_np),packed,out_features=N,in_features=K))
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-4)
