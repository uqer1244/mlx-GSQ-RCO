from pathlib import Path
import pytest

def test_public_load_returns_mlx_lm_compatible_pair():
    pytest.importorskip("mlx.core"); pytest.importorskip("mlx_lm")
    from mlx_gsq import load
    p=Path("Qwen3.8-27B-GSQ-RCO/model-packed.safetensors")
    tok=p.parent/"tokenizer"
    if not p.exists() or not tok.exists(): pytest.skip("local model assets absent")
    model,tokenizer=load(p)
    assert len(model.layers)==64
    assert tokenizer.encode("Hello")==[9419]
