# mlx-GSQ-RCO

Run the GSQ-RCO mixed-IQ quantization of Qwen3.8-27B on Apple Silicon using
MLX and custom Metal kernels.

This project preserves the original GGUF quantized bytes inside a Safetensors
container and routes every tensor to a decoder/QMV kernel for its GGUF type.
It does not dequantize the full model into FP16 and does not requantize the
weights into MLX-LM's standard uniform format.

> Status: alpha. This runtime currently targets the Qwen3.8-27B GSQ-RCO
> checkpoint and Apple Silicon.

## Model weights

Weights are intentionally not stored in this GitHub repository.

- MLX checkpoint:
  [`uqer1244/Qwen3.8-27B-GSQ-RCO-MLX`](https://huggingface.co/uqer1244/Qwen3.8-27B-GSQ-RCO-MLX)
- Direct GGUF parent:
  [`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)
- Original base model:
  [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)

The Hugging Face MLX repository contains `model.safetensors`, its manifest,
and tokenizer/configuration files.

## Supported tensor formats

The runtime supports all 11 quantized formats present in the checkpoint:

```text
IQ1_M  IQ1_S  IQ2_XXS  IQ2_XS  IQ2_S  IQ3_XXS
IQ3_S  IQ4_XS Q2_K     Q4_K    Q6_K
```

F32 and BF16 tensors are reinterpreted directly without numeric conversion.

## Requirements

- Apple Silicon Mac with Metal
- Python 3.10+
- approximately 10.4 GB of unified memory for a minimal generation run;
  additional headroom is recommended

## Install

Until a PyPI package is published:

```bash
git clone https://github.com/uqer1244/mlx-GSQ-RCO.git
cd mlx-GSQ-RCO
python -m pip install -e '.[mlx,integration]'
```

For development and reference tests:

```bash
python -m pip install -e '.[mlx,integration,dev]'
```

## Download and run

```python
from huggingface_hub import snapshot_download
from mlx_gsq import load
from mlx_lm.generate import generate

model_dir = snapshot_download(
    "uqer1244/Qwen3.8-27B-GSQ-RCO-MLX"
)
model, tokenizer = load(
    f"{model_dir}/model.safetensors",
    tokenizer_path=model_dir,
)

text = generate(model, tokenizer, prompt="Hello", max_tokens=32)
print(text)
```

CLI usage after downloading the model repository:

```bash
mlx-gsq-generate \
  uqer1244/Qwen3.8-27B-GSQ-RCO-MLX \
  --prompt "Hello" \
  --max-tokens 32 \
  --verbose
```

The CLI accepts a Hugging Face model ID, a local model directory, or a direct
path to `model.safetensors`.

`mlx_lm.load()` cannot load this checkpoint directly. Use `mlx_gsq.load()` so
the packed tensor metadata is routed to the custom Metal kernels.

## Convert the parent GGUF yourself

Conversion is lossless and streaming. Each tensor's original packed payload is
copied into a U8 Safetensors tensor; no quantization values are changed.

```bash
mlx-gsq-gguf analyze model.gguf -o analysis.json

mlx-gsq-gguf convert \
  model.gguf \
  model.safetensors

mlx-gsq-gguf verify \
  model.gguf \
  model.safetensors
```

Conversion produces:

```text
model.safetensors
model.safetensors.manifest.json
```

The verifier compares the SHA-256 digest of all 866 packed tensor payloads
against the GGUF source.

## Architecture

```text
mixed-IQ GGUF
    │ lossless streaming conversion
    ▼
packed U8 Safetensors + manifest
    │ tensor qtype routing
    ▼
MLX custom Metal decode / fused QMV
    │
    ▼
MLX-LM Qwen3.5 hybrid model adapter
```

The packed format is identified by the Safetensors metadata value
`format=mlx-gsq-packed-v1`.

## Current limitations

- Prefill is functional through the multi-row QMV path but does not yet have a
  dedicated tiled QMM kernel.
- The current adapter targets this Qwen3.8/Qwen3.5 hybrid architecture.
- Vision inference is not implemented; only text generation is supported.
- This is a custom packed format, not a drop-in standard MLX-LM checkpoint.

## License and attribution

Code in this repository is released under the MIT License. The model weights
are distributed separately under their upstream Apache-2.0 license and retain
their upstream attribution.
See [NOTICE](NOTICE) and the upstream model cards before redistribution.
