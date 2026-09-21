"""Decode-only Metal kernels for K-quants and IQ4_XS.

These kernels are correctness building blocks. Inference uses the same unpack
logic fused into QMV/QMM so decoded matrices are never materialized.
"""
from __future__ import annotations

from functools import lru_cache


_COMMON_START = r"""
    uint index = thread_position_in_grid.x;
    uint block_index = index / 256u;
    uint element = index - block_index * 256u;
"""

_SOURCES = {
    "Q2_K": _COMMON_START + r"""
    device const uchar* b = packed + block_index * 84u;
    uint chunk = element / 128u;
    uint in_chunk = element - chunk * 128u;
    uint plane = in_chunk / 32u;
    uint pos = in_chunk - plane * 32u;
    uint q = (uint(b[16u + chunk * 32u + pos]) >> (2u * plane)) & 3u;
    uint group = element / 16u;
    uchar sm = b[group];
    float d = float(*(reinterpret_cast<device const half*>(b + 80u)));
    float dmin = float(*(reinterpret_cast<device const half*>(b + 82u)));
    output[index] = d * float(sm & 15u) * float(q) - dmin * float(sm >> 4u);
    """,
    "Q4_K": _COMMON_START + r"""
    device const uchar* b = packed + block_index * 144u;
    device const uchar* s = b + 4u;
    uint group = element / 32u;
    uint pos = element - group * 32u;
    uint qbyte = uint(b[16u + (group / 2u) * 32u + pos]);
    uint q = (qbyte >> ((group & 1u) * 4u)) & 15u;
    uint sc, mn;
    if (group < 4u) {
        sc = uint(s[group]) & 63u;
        mn = uint(s[4u + group]) & 63u;
    } else {
        uint g = group - 4u;
        sc = (uint(s[8u + g]) & 15u) | ((uint(s[g]) >> 2u) & 48u);
        mn = (uint(s[8u + g]) >> 4u) | ((uint(s[4u + g]) >> 2u) & 48u);
    }
    float d = float(*(reinterpret_cast<device const half*>(b)));
    float dmin = float(*(reinterpret_cast<device const half*>(b + 2u)));
    output[index] = d * float(sc) * float(q) - dmin * float(mn);
    """,
    "Q6_K": _COMMON_START + r"""
    device const uchar* b = packed + block_index * 210u;
    uint chunk = element / 128u;
    uint in_chunk = element - chunk * 128u;
    uint ql_half = in_chunk / 64u;
    uint ql_pos = in_chunk - ql_half * 64u;
    uint ql = (uint(b[chunk * 64u + ql_pos]) >> (4u * ql_half)) & 15u;
    uint qh_plane = in_chunk / 32u;
    uint qh_pos = in_chunk - qh_plane * 32u;
    uint qh = (uint(b[128u + chunk * 32u + qh_pos]) >> (2u * qh_plane)) & 3u;
    int q = int(ql | (qh << 4u)) - 32;
    int scale = int(reinterpret_cast<device const char*>(b + 192u)[element / 16u]);
    float d = float(*(reinterpret_cast<device const half*>(b + 208u)));
    output[index] = d * float(scale) * float(q);
    """,
    "IQ4_XS": _COMMON_START + r"""
    device const uchar* b = packed + block_index * 136u;
    uint group = element / 32u;
    uint pos = element - group * 32u;
    uint scales_h = uint(b[2]) | (uint(b[3]) << 8u);
    uint low_byte = uint(b[4u + group / 2u]);
    uint low = (low_byte >> (4u * (group & 1u))) & 15u;
    uint high = (scales_h >> (2u * group)) & 3u;
    int scale = int(low | (high << 4u)) - 32;
    uint qbyte = uint(b[8u + group * 16u + (pos & 15u)]);
    uint q = (qbyte >> (4u * (pos / 16u))) & 15u;
    constexpr int values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    float d = float(*(reinterpret_cast<device const half*>(b)));
    output[index] = d * float(scale) * float(values[q]);
    """,
}

_BLOCK_BYTES = {"Q2_K": 84, "Q4_K": 144, "Q6_K": 210, "IQ4_XS": 136}


@lru_cache(maxsize=None)
def _kernel(qtype: str):
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name=f"mlx_gsq_decode_{qtype.lower()}",
        input_names=["packed"], output_names=["output"], source=_SOURCES[qtype]
    )


def decode_k_quant(packed, qtype: str):
    import mlx.core as mx
    if qtype not in _SOURCES: raise KeyError(qtype)
    if packed.dtype != mx.uint8: raise TypeError("packed input must be uint8")
    block_bytes = _BLOCK_BYTES[qtype]
    if packed.size % block_bytes: raise ValueError(f"{qtype} payload is not block aligned")
    size = packed.size // block_bytes * 256
    return _kernel(qtype)(
        inputs=[packed], grid=(size,1,1), threadgroup=(256,1,1),
        output_shapes=[(size,)], output_dtypes=[mx.float32]
    )[0]
