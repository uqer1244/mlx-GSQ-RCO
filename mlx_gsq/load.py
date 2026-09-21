"""Public model loader."""
from pathlib import Path
import mlx.core as mx
from .runtime import PackedModelStore
from .models import build_qwen35


def resolve_model_path(path: str|Path) -> Path:
    """Resolve a local Safetensors path, local model directory, or HF repo ID."""
    value=str(path)
    local=Path(value).expanduser()
    if local.is_file():
        return local
    if local.is_dir():
        model=local/"model.safetensors"
        if not model.is_file():
            raise FileNotFoundError(f"model.safetensors not found in {local}")
        return model
    if "/" in value:
        from huggingface_hub import snapshot_download
        directory=Path(snapshot_download(
            repo_id=value,
            allow_patterns=["*.safetensors","*.json","*.jinja","*.txt"],
        ))
        model=directory/"model.safetensors"
        if not model.is_file():
            raise FileNotFoundError(f"model.safetensors not found in {value}")
        return model
    raise FileNotFoundError(value)


def load_model(path: str|Path,dtype=mx.float16):
    path=resolve_model_path(path)
    store=PackedModelStore(path)
    if store.metadata.get("architecture")!="qwen35": raise ValueError("only qwen35 is currently supported")
    return build_qwen35(store,dtype=dtype)


def load(path: str|Path,tokenizer_path: str|Path|None=None,dtype=mx.float16):
    """Load a packed model and an mlx-lm compatible tokenizer."""
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    path=resolve_model_path(path)
    if tokenizer_path is None:
        nested=path.parent/"tokenizer"
        tokenizer_path=nested if nested.is_dir() else path.parent
    else:
        tokenizer_path=Path(tokenizer_path)
    return load_model(path,dtype=dtype),load_tokenizer(tokenizer_path)
