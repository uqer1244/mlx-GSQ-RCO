"""Fused IQ3_S decode + matrix-vector/matrix multiplication."""
from __future__ import annotations
from functools import lru_cache

_SOURCE=r"""
    uint tid = thread_index_in_threadgroup;
    uint out_row = threadgroup_position_in_grid.x;
    uint input_row = threadgroup_position_in_grid.y;
    uint K = shape[0];
    uint N = shape[1];
    uint blocks_per_row = K / 256u;
    float sum = 0.0f;

    for (uint bc = 0; bc < blocks_per_row; ++bc) {
        device const uchar* block = packed + (out_row * blocks_per_row + bc) * 110u;
        uint group = tid / 32u;
        uint within_group = tid - group * 32u;
        uint lane = within_group / 8u;
        uint component = within_group - lane * 8u;
        float d = float(*(reinterpret_cast<device const half*>(block)));
        device const uchar* qs = block + 2u;
        device const uchar* qh = block + 66u;
        device const uchar* signs = block + 74u;
        device const uchar* scales = block + 106u;
        uint scale_byte = uint(scales[group >> 1u]);
        uint scale = (scale_byte >> (4u * (group & 1u))) & 15u;
        float db = d * float(1u + 2u * scale);
        uint qbase = group * 8u + lane * 2u;
        uint gi, gc;
        if (component < 4u) {
            gi = uint(qs[qbase]) | ((uint(qh[group]) << (8u - 2u * lane)) & 256u);
            gc = component;
        } else {
            gi = uint(qs[qbase + 1u]) | ((uint(qh[group]) << (7u - 2u * lane)) & 256u);
            gc = component - 4u;
        }
        float w = db * grid[gi * 4u + gc];
        if ((uint(signs[group * 4u + lane]) & (1u << component)) != 0u) w = -w;
        sum += w * float(x[input_row * K + bc * 256u + tid]);
    }

    threadgroup float partial[256];
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) output[input_row * N + out_row] = partial[0];
"""


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name="mlx_gsq_iq3_s_qmv", input_names=["x","packed","grid","shape"],
        output_names=["output"], source=_SOURCE,
    )


def iq3_s_qmv(x, packed, *, out_features: int, in_features: int):
    import mlx.core as mx
    from .iq3_s_decode import _grid
    if in_features % 256: raise ValueError("IQ3_S in_features must be divisible by 256")
    expected=out_features*(in_features//256)*110
    if packed.size != expected: raise ValueError(f"packed size {packed.size} != expected {expected}")
    original=x.shape[:-1]
    if x.shape[-1] != in_features: raise ValueError("input feature size mismatch")
    rows=x.size//in_features; flat=x.reshape((rows,in_features))
    shape=mx.array([in_features,out_features],dtype=mx.uint32)
    out=_kernel()(inputs=[flat,packed,_grid(),shape],grid=(out_features*256,rows,1),threadgroup=(256,1,1),output_shapes=[(rows,out_features)],output_dtypes=[mx.float32])[0]
    return out.reshape((*original,out_features))
