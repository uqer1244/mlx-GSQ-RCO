"""IQ3_S block decoder implemented as an MLX custom Metal kernel."""
from __future__ import annotations

from functools import lru_cache


_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    uint block_index = index / 256u;
    uint element = index - block_index * 256u;
    uint group = element / 32u;
    uint within_group = element - group * 32u;
    uint lane = within_group / 8u;
    uint component = within_group - lane * 8u;

    device const uchar* block = packed + block_index * 110u;
    half d_half = *(reinterpret_cast<device const half*>(block));
    float d = float(d_half);
    device const uchar* qs = block + 2u;
    device const uchar* qh = block + 66u;
    device const uchar* signs = block + 74u;
    device const uchar* scales = block + 106u;

    uchar scale_byte = scales[group >> 1u];
    uint scale_nibble = (group & 1u) ? uint(scale_byte >> 4u) : uint(scale_byte & 15u);
    float db = d * float(1u + 2u * scale_nibble);

    uint qbase = group * 8u + lane * 2u;
    uint grid_index;
    uint grid_component;
    if (component < 4u) {
        grid_index = uint(qs[qbase]) | ((uint(qh[group]) << (8u - 2u * lane)) & 256u);
        grid_component = component;
    } else {
        grid_index = uint(qs[qbase + 1u]) | ((uint(qh[group]) << (7u - 2u * lane)) & 256u);
        grid_component = component - 4u;
    }
    float magnitude = grid[grid_index * 4u + grid_component];
    bool negative = (uint(signs[group * 4u + lane]) & (1u << component)) != 0u;
    output[index] = negative ? -db * magnitude : db * magnitude;
"""


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlx_gsq_decode_iq3_s",
        input_names=["packed", "grid"],
        output_names=["output"],
        source=_SOURCE,
    )


@lru_cache(maxsize=1)
def _grid():
    import mlx.core as mx
    from mlx_gsq.quant import iq3_s_grid

    return mx.array(iq3_s_grid().reshape(-1), dtype=mx.float32)


def decode_iq3_s(packed):
    """Decode a flat U8 IQ3_S payload into a flat MLX float32 array."""
    import mlx.core as mx

    if packed.dtype != mx.uint8:
        raise TypeError(f"IQ3_S packed input must be uint8, got {packed.dtype}")
    if packed.size % 110:
        raise ValueError(f"IQ3_S payload size must be divisible by 110, got {packed.size}")
    output_size = packed.size // 110 * 256
    return _kernel()(
        inputs=[packed, _grid()],
        grid=(output_size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(output_size,)],
        output_dtypes=[mx.float32],
    )[0]
