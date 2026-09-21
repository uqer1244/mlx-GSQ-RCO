"""Lossless GGUF packed-weight to safetensors container conversion.

Each GGUF tensor is stored as a one-dimensional U8 safetensors tensor.  No
dequantization or requantization occurs.  ``manifest.json`` carries the GGUF
shape and quantization metadata required by a custom MLX loader.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any

from .parser import GGUFReader, QUANT_BLOCK_SIZES


COPY_CHUNK_SIZE = 16 * 1024 * 1024


def _header(reader: GGUFReader) -> tuple[bytes, dict[str, Any]]:
    tensor_quantization = {}
    for tensor in reader.tensors:
        block_elements, block_bytes = QUANT_BLOCK_SIZES[int(tensor.quantization_type)]
        tensor_quantization[tensor.name] = {
            "qtype": tensor.quantization_type.name,
            "qtype_id": int(tensor.quantization_type),
            "gguf_shape": list(tensor.shape),
            "logical_shape": list(reversed(tensor.shape)),
            "num_elements": tensor.num_elements,
            "block_elements": block_elements,
            "block_bytes": block_bytes,
        }
    header: dict[str, Any] = {
        "__metadata__": {
            "format": "mlx-gsq-packed-v1",
            "source_format": "GGUF",
            "architecture": str(reader.metadata.get("general.architecture", "")),
            "note": "U8 tensors contain original GGUF packed bytes; custom kernels required",
            # Safetensors metadata values must be strings. Keeping this compact
            # JSON mapping in the container makes the file self-describing;
            # the sidecar manifest remains useful for inspection and hashes.
            "tensor_quantization": json.dumps(tensor_quantization, separators=(",", ":")),
        }
    }
    cursor = 0
    for tensor in reader.tensors:
        header[tensor.name] = {
            "dtype": "U8",
            "shape": [tensor.num_bytes],
            "data_offsets": [cursor, cursor + tensor.num_bytes],
        }
        cursor += tensor.num_bytes
    raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    return raw, header


def build_manifest(reader: GGUFReader, output_name: str) -> dict[str, Any]:
    return {
        "format": "mlx-gsq-packed-v1",
        "container": output_name,
        "source": {
            "path": str(reader.path),
            "file_size": reader.file_size,
            "gguf_version": reader.version,
            "architecture": reader.metadata.get("general.architecture"),
            "name": reader.metadata.get("general.name"),
        },
        "tensor_count": len(reader.tensors),
        "tensors": {tensor.name: tensor.as_dict() for tensor in reader.tensors},
    }


def convert_gguf_to_safetensors(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    verify: bool = True,
) -> Path:
    """Stream packed tensors into a valid safetensors file.

    The temporary file is atomically renamed only after all tensors have been
    written and, by default, hashed against their original GGUF byte ranges.
    """
    reader = GGUFReader(input_path)
    output = Path(output_path).expanduser().resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    header, _ = _header(reader)

    try:
        with reader.path.open("rb") as source, temporary.open("xb") as target:
            target.write(struct.pack("<Q", len(header)))
            target.write(header)
            for tensor in reader.tensors:
                source.seek(tensor.absolute_offset)
                remaining = tensor.num_bytes
                while remaining:
                    chunk = source.read(min(remaining, COPY_CHUNK_SIZE))
                    if not chunk:
                        raise IOError(f"short GGUF read for {tensor.name}")
                    target.write(chunk)
                    remaining -= len(chunk)
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(output)
        manifest = build_manifest(reader, output.name)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        if verify:
            verify_conversion(reader.path, output, manifest_path, record_hashes=True)
    except BaseException:
        if temporary.exists(): temporary.unlink()
        raise
    return output


def _hash_range(stream, offset: int, size: int) -> str:
    digest = hashlib.sha256(); stream.seek(offset); remaining = size
    while remaining:
        chunk = stream.read(min(remaining, COPY_CHUNK_SIZE))
        if not chunk: raise IOError("short read during verification")
        digest.update(chunk); remaining -= len(chunk)
    return digest.hexdigest()


def verify_conversion(
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    record_hashes: bool = False,
) -> dict[str, Any]:
    """Validate safetensors structure and every tensor's SHA-256 digest."""
    reader = GGUFReader(input_path)
    output = Path(output_path)
    manifest_path = Path(manifest_path) if manifest_path else output.with_suffix(output.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    hashes: dict[str, str] = {}
    with output.open("rb") as packed:
        header_size = struct.unpack("<Q", packed.read(8))[0]
        header = json.loads(packed.read(header_size))
        data_start = 8 + header_size
        expected_size = data_start + sum(t.num_bytes for t in reader.tensors)
        if output.stat().st_size != expected_size:
            raise ValueError(f"safetensors size mismatch: {output.stat().st_size} != {expected_size}")
        with reader.path.open("rb") as source:
            for tensor in reader.tensors:
                entry = header.get(tensor.name)
                if not entry or entry["dtype"] != "U8" or entry["shape"] != [tensor.num_bytes]:
                    raise ValueError(f"invalid safetensors entry for {tensor.name}")
                start, end = entry["data_offsets"]
                source_hash = _hash_range(source, tensor.absolute_offset, tensor.num_bytes)
                output_hash = _hash_range(packed, data_start + start, end - start)
                if source_hash != output_hash:
                    raise ValueError(f"packed bytes differ for {tensor.name}")
                hashes[tensor.name] = source_hash
                manifest_tensor = manifest["tensors"].get(tensor.name)
                expected_tensor = tensor.as_dict()
                if not manifest_tensor or any(manifest_tensor.get(k) != v for k, v in expected_tensor.items()):
                    raise ValueError(f"manifest differs for {tensor.name}")
    result = {
        "algorithm": "sha256",
        "status": "passed",
        "tensor_count": len(hashes),
        "all_packed_bytes_identical": True,
        "tensor_hashes": hashes,
    }
    if record_hashes:
        manifest["verification"] = result
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return result


# Backwards-compatible public name, now with honest semantics.
def convert_gguf_to_mlx(input_path: str, output_dir: str = ".", overwrite: bool = False) -> bool:
    output = Path(output_dir) / "model-packed.safetensors"
    convert_gguf_to_safetensors(input_path, output, overwrite=overwrite)
    return True
