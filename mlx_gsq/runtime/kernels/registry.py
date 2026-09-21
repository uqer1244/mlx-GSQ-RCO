"""Decode kernel registry. Only validated kernels are registered."""
from __future__ import annotations

from .iq3_s_decode import decode_iq3_s
from .k_decode import decode_k_quant
from .iq_lookup_decode import decode_lookup_quant

SUPPORTED_QUANT_TYPES = (
    "IQ1_M", "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS",
    "IQ3_S", "IQ4_XS", "Q2_K", "Q4_K", "Q6_K",
)

DECODERS = {
    "IQ1_M": lambda x: decode_lookup_quant(x, "IQ1_M"),
    "IQ1_S": lambda x: decode_lookup_quant(x, "IQ1_S"),
    "IQ2_XXS": lambda x: decode_lookup_quant(x, "IQ2_XXS"),
    "IQ2_XS": lambda x: decode_lookup_quant(x, "IQ2_XS"),
    "IQ2_S": lambda x: decode_lookup_quant(x, "IQ2_S"),
    "IQ3_XXS": lambda x: decode_lookup_quant(x, "IQ3_XXS"),
    "IQ3_S": decode_iq3_s,
    "IQ4_XS": lambda x: decode_k_quant(x, "IQ4_XS"),
    "Q2_K": lambda x: decode_k_quant(x, "Q2_K"),
    "Q4_K": lambda x: decode_k_quant(x, "Q4_K"),
    "Q6_K": lambda x: decode_k_quant(x, "Q6_K"),
}


def decode(packed, qtype: str):
    try: decoder = DECODERS[qtype]
    except KeyError as error:
        raise NotImplementedError(f"validated Metal decoder not yet available for {qtype}") from error
    return decoder(packed)


def decode_weight(weight, *, block_count: int | None = None):
    """Route a PackedWeight to its validated decoder using embedded metadata."""
    from mlx_gsq.gguf.parser import QUANT_BLOCK_SIZES

    qtype = weight.quantization_type
    if qtype not in SUPPORTED_QUANT_TYPES:
        raise TypeError(f"{weight.name} is not a supported quantized weight: {qtype}")
    data = weight.data
    if block_count is not None:
        block_bytes = QUANT_BLOCK_SIZES[weight.quantization_type_id][1]
        data = data[: block_count * block_bytes]
    return decode(data, qtype)
