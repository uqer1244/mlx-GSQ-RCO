"""Public model loader."""
from pathlib import Path
import mlx.core as mx
from .runtime import PackedModelStore
from .models import build_qwen35


def load_model(path: str|Path,dtype=mx.float16):
    store=PackedModelStore(path)
    if store.metadata.get("architecture")!="qwen35": raise ValueError("only qwen35 is currently supported")
    return build_qwen35(store,dtype=dtype)


def load(path: str|Path,tokenizer_path: str|Path|None=None,dtype=mx.float16):
    """Load a packed model and an mlx-lm compatible tokenizer."""
    from mlx_lm.tokenizer_utils import load as load_tokenizer
    path=Path(path)
    tokenizer_path=Path(tokenizer_path) if tokenizer_path else path.parent/"tokenizer"
    return load_model(path,dtype=dtype),load_tokenizer(tokenizer_path)
