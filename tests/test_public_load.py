from pathlib import Path

import pytest

from mlx_gsq.load import resolve_model_path

def test_public_load_returns_mlx_lm_compatible_pair():
    pytest.importorskip("mlx.core"); pytest.importorskip("mlx_lm")
    from mlx_gsq import load
    p=Path("Qwen3.8-27B-GSQ-RCO/model-packed.safetensors")
    tok=p.parent/"tokenizer"
    if not p.exists() or not tok.exists(): pytest.skip("local model assets absent")
    model,tokenizer=load(p)
    assert len(model.layers)==64
    assert tokenizer.encode("Hello")==[9419]


def test_resolve_model_directory(tmp_path: Path):
    model=tmp_path/"model.safetensors"
    model.touch()
    assert resolve_model_path(tmp_path)==model


def test_resolve_missing_local_name():
    with pytest.raises(FileNotFoundError):
        resolve_model_path("missing-local-model")
