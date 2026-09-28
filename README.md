# mlx-GSQ-RCO

Run the mixed-IQ GSQ-RCO quantization of Qwen3.8-27B on Apple Silicon with
MLX and custom Metal kernels. The runtime keeps the original GGUF quantized
payloads packed in Safetensors; it does not expand the full model to FP16 or
requantize it to a uniform MLX format.

**Status: alpha.** The released runtime targets this Qwen3.8-27B checkpoint on
Apple Silicon. Text generation is supported; vision inference is not.

## Model

Download the packed checkpoint from
[`uqer1244/Qwen3.8-27B-GSQ-RCO-MLX`](https://huggingface.co/uqer1244/Qwen3.8-27B-GSQ-RCO-MLX).
The original mixed-IQ GGUF is
[`ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF).
Weights are hosted on Hugging Face and are not included in this code repository.

The runtime supports the checkpoint's 11 quantized types: `IQ1_M`, `IQ1_S`,
`IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS`, `Q2_K`, `Q4_K`,
and `Q6_K`. Plain F32/BF16 tensors are loaded without numeric conversion.

## Requirements

- Apple Silicon Mac with Metal
- Python 3.10+
- About 10.4 GB unified memory for a minimal generation run; extra headroom is
  recommended

## Install

```bash
python -m pip install \
  'mlx-gsq-rco[mlx,integration] @ git+https://github.com/uqer1244/mlx-GSQ-RCO.git'
```

For a local checkout:

```bash
git clone https://github.com/uqer1244/mlx-GSQ-RCO.git
cd mlx-GSQ-RCO
python -m pip install -e '.[mlx,integration]'
```

## Generate text

The CLI downloads the model from Hugging Face on first use:

```bash
mlx-gsq-generate \
  uqer1244/Qwen3.8-27B-GSQ-RCO-MLX \
  --prompt "Hello" \
  --max-tokens 32
```

Python API:

```python
from mlx_gsq import load
from mlx_lm.generate import generate

model, tokenizer = load("uqer1244/Qwen3.8-27B-GSQ-RCO-MLX")
print(generate(model, tokenizer, prompt="Hello", max_tokens=32))
```

Use `mlx_gsq.load()` for this custom packed format. `mlx_lm.load()` does not
load it directly.

## Performance

Measurements below were taken on an Apple M3 Pro and depend on prompt length,
software versions, and runtime conditions.

| Measurement | MLX-GSQ-RCO | llama.cpp Metal |
| --- | ---: | ---: |
| 64-token decode | 9.14 tok/s | 9.78 tok/s |
| 182-token prompt prefill | 22.3 tok/s | 83.05 tok/s |
| 256-token generation | 9.03 tok/s | — |
| Peak unified memory during generation | about 10.3 GB | — |

The tiled QMM prefill kernel reuses each decoded weight tile across input rows.
Its half4 vectorized GEMM measured 2.6–3.2x faster than the earlier scalar
GEMM kernel on large single-layer cases; end-to-end prefill improved from
10.5 to 22.3 tok/s in the recorded run. The benchmark scripts are in `scripts/`.

## Convert the source GGUF

Conversion streams the original packed payloads into Safetensors without
changing the quantized bytes:

```bash
mlx-gsq-gguf analyze model.gguf -o analysis.json
mlx-gsq-gguf convert model.gguf model.safetensors
mlx-gsq-gguf verify model.gguf model.safetensors
```

The verifier compares SHA-256 hashes for all 866 packed tensor payloads.

## Limits

- The model adapter targets the Qwen3.8/Qwen3.5 hybrid checkpoint.
- Only text generation is implemented; vision inference is not supported.
- The packed Safetensors file is a custom format and needs this runtime.
- Performance results are from one Apple Silicon system and are not a promise
  for other machines.

## License and attribution

Runtime code is MIT licensed. The separately hosted model weights retain the
upstream Apache-2.0 license and attribution. See [NOTICE](NOTICE) and the
upstream model cards.
