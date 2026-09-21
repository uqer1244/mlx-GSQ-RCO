from pathlib import Path
import numpy as np
import pytest
from mlx_gsq.quant import dequantize_reference

def test_iq2_s_embedding_selected_rows():
    mx=pytest.importorskip("mlx.core")
    if not mx.metal.is_available(): pytest.skip("Metal unavailable")
    from mlx_gsq.runtime import PackedModelStore
    from mlx_gsq.runtime.kernels.embedding import iq2_s_embedding
    p=Path("Qwen3.8-27B-GSQ-RCO/model-packed.safetensors")
    if not p.exists(): pytest.skip("packed model absent")
    s=PackedModelStore(p); w=s.weight("token_embd.weight"); ids_np=np.array([[0,17,248046]],dtype=np.int32)
    row_bytes=w.num_bytes//w.logical_shape[0]; expected=[]
    for token in ids_np.reshape(-1): expected.append(dequantize_reference(np.asarray(w.data[token*row_bytes:(token+1)*row_bytes]),"IQ2_S"))
    expected=np.stack(expected).reshape((*ids_np.shape,w.logical_shape[1])).astype(np.float16)
    actual=np.asarray(iq2_s_embedding(mx.array(ids_np),w.data,vocab_size=w.logical_shape[0],dimensions=w.logical_shape[1],dtype=mx.float16))
    np.testing.assert_array_equal(actual,expected)
