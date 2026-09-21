from pathlib import Path
import pytest


def test_restore_a_log_round_trip():
    mx=pytest.importorskip("mlx.core")
    from mlx_gsq.models.qwen35 import _restore_a_log
    ssm_a=mx.array([-0.25,-0.125,-0.0625],dtype=mx.float32)
    assert mx.allclose(-mx.exp(_restore_a_log(ssm_a)),ssm_a)

def test_qwen35_adapter_replaces_all_large_weights():
    mx=pytest.importorskip("mlx.core"); pytest.importorskip("mlx_lm")
    from mlx_gsq import load_model
    from mlx_gsq.nn import GSQEmbedding,GSQLinear
    p=Path("Qwen3.8-27B-GSQ-RCO/model-packed.safetensors")
    if not p.exists(): pytest.skip("packed model absent")
    model=load_model(p)
    assert isinstance(model.model.embed_tokens,GSQEmbedding)
    assert isinstance(model.lm_head,GSQLinear)
    assert len(model.layers)==64
    for layer in model.layers:
        assert isinstance(layer.mlp.gate_proj,GSQLinear)
        assert isinstance(layer.mlp.up_proj,GSQLinear)
        assert isinstance(layer.mlp.down_proj,GSQLinear)
        if layer.is_linear:
            assert type(layer.linear_attn).__name__=="GSQGatedDeltaNet"
            assert isinstance(layer.linear_attn.in_proj_qkv,GSQLinear)
            assert isinstance(layer.linear_attn.in_proj_z,GSQLinear)
            assert isinstance(layer.linear_attn.out_proj,GSQLinear)
        else:
            assert isinstance(layer.self_attn.q_proj,GSQLinear)
            assert isinstance(layer.self_attn.k_proj,GSQLinear)
            assert isinstance(layer.self_attn.v_proj,GSQLinear)
            assert isinstance(layer.self_attn.o_proj,GSQLinear)
