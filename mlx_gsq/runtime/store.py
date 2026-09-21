"""Load an mlx-gsq packed safetensors container with MLX."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PackedWeight:
    name: str
    data: Any
    gguf_shape: tuple[int, ...]
    logical_shape: tuple[int, ...]
    quantization_type: str
    quantization_type_id: int
    num_elements: int
    num_bytes: int

    @property
    def is_quantized(self) -> bool:
        return self.quantization_type not in {"F32", "F16", "BF16"}


class PackedModelStore:
    """MLX-backed view of packed weights and their GGUF metadata."""

    def __init__(self, model_path: str | Path, manifest_path: str | Path | None = None):
        try:
            import mlx.core as mx
        except ImportError as error:
            raise RuntimeError("MLX is required; install with `uv sync --extra mlx`") from error
        self.mx = mx
        self.path = Path(model_path).expanduser().resolve()
        arrays, metadata = mx.load(str(self.path), return_metadata=True)
        if metadata.get("format") != "mlx-gsq-packed-v1":
            raise ValueError("safetensors metadata does not describe an mlx-gsq model")
        manifest_file = Path(manifest_path) if manifest_path else self.path.with_suffix(self.path.suffix + ".manifest.json")
        if manifest_file.exists():
            self.manifest = json.loads(manifest_file.read_text())
        else:
            embedded = json.loads(metadata["tensor_quantization"])
            self.manifest = {
                "format": "mlx-gsq-packed-v1",
                "tensors": {
                    name: {
                        "name": name,
                        "gguf_shape": info["gguf_shape"],
                        "logical_shape": info["logical_shape"],
                        "quantization_type": info["qtype"],
                        "quantization_type_id": info["qtype_id"],
                        "num_elements": info["num_elements"],
                        "num_bytes": int(arrays[name].size),
                    }
                    for name, info in embedded.items()
                },
            }
        if self.manifest.get("format") != "mlx-gsq-packed-v1":
            raise ValueError("unsupported packed model format")
        expected = self.manifest["tensors"]
        if set(arrays) != set(expected):
            raise ValueError("manifest and safetensors tensor names differ")
        self.arrays = arrays
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.arrays)

    def __contains__(self, name: str) -> bool:
        return name in self.arrays

    def weight(self, name: str) -> PackedWeight:
        info = self.manifest["tensors"][name]
        data = self.arrays[name]
        if data.dtype != self.mx.uint8 or data.size != info["num_bytes"]:
            raise ValueError(f"invalid packed data for {name}")
        return PackedWeight(
            name=name,
            data=data,
            gguf_shape=tuple(info["gguf_shape"]),
            logical_shape=tuple(info["logical_shape"]),
            quantization_type=info["quantization_type"],
            quantization_type_id=info["quantization_type_id"],
            num_elements=info["num_elements"],
            num_bytes=info["num_bytes"],
        )

    def materialize_plain(self, name: str):
        """Interpret an unquantized F32/F16/BF16 tensor without numeric conversion."""
        weight = self.weight(name)
        dtypes = {"F32": self.mx.float32, "F16": self.mx.float16, "BF16": self.mx.bfloat16}
        if weight.quantization_type not in dtypes:
            raise TypeError(f"{name} is {weight.quantization_type}; a quantized kernel is required")
        return self.mx.view(weight.data, dtypes[weight.quantization_type]).reshape(weight.logical_shape)
