"""Generate fused decode + QMV kernels from validated decoder bodies."""
from __future__ import annotations
from functools import lru_cache
import re

from .iq3_s_decode import _SOURCE as _IQ3_SOURCE
from .iq_lookup_decode import _SOURCES as _IQ_SOURCES, _tables
from .k_decode import _SOURCES as _K_SOURCES


def _decoder_body(qtype: str) -> str:
    source = _IQ3_SOURCE if qtype == "IQ3_S" else (_IQ_SOURCES.get(qtype) or _K_SOURCES[qtype])
    source = re.sub(r"uint\s+index\s*=\s*thread_position_in_grid\.x\s*;", "", source, count=1)
    match = list(re.finditer(r"output\s*\[\s*index\s*\]\s*=\s*([^;]+);", source))
    if len(match) != 1:
        raise RuntimeError(f"cannot derive QMV decoder body for {qtype}")
    m = match[0]
    replacement = "float decoded_weight = " + m.group(1) + ";\n        sum += decoded_weight * float(x[input_row * K + bc * 256u + tid]);"
    return source[:m.start()] + replacement + source[m.end():]


def _source(qtype: str, in_features: int, out_features: int) -> str:
    body = _decoder_body(qtype)
    return f"""
    uint tid=thread_index_in_threadgroup;
    uint out_row=threadgroup_position_in_grid.x;
    uint input_row=threadgroup_position_in_grid.y;
    constexpr uint K={in_features}u;
    constexpr uint N={out_features}u;
    constexpr uint blocks_per_row=K/256u;
    float sum=0.0f;
    for(uint bc=0;bc<blocks_per_row;++bc){{
        uint index=(out_row*blocks_per_row+bc)*256u+tid;
    """ + body + r"""
    }
    // Reduce within SIMD groups first. This replaces eight full-threadgroup
    // barrier/reduction rounds with one SIMD reduction and one small
    // cross-SIMD reduction.
    threadgroup float partial[8];
    float simd_total=simd_sum(sum);
    if(thread_index_in_simdgroup==0u)
        partial[simdgroup_index_in_threadgroup]=simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(simdgroup_index_in_threadgroup==0u){
        float group_total=thread_index_in_simdgroup<8u
            ? partial[thread_index_in_simdgroup] : 0.0f;
        group_total=simd_sum(group_total);
        if(thread_index_in_simdgroup==0u)
            output[input_row*N+out_row]=group_total;
    }
    """


@lru_cache(maxsize=None)
def _kernel(qtype: str, in_features: int, out_features: int):
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name=f"mlx_gsq_{qtype.lower()}_{in_features}_{out_features}_qmv",
        input_names=["x","packed","grid","ksigns"],
        output_names=["output"],source=_source(qtype, in_features, out_features),
    )


def _iq3_s_source(in_features: int, out_features: int) -> str:
    return f"""
    constexpr uint K={in_features}u;
    constexpr uint N={out_features}u;
    constexpr uint blocks_per_row=K/256u;
    constexpr uint groups_per_row=K/32u;
    uint lane=thread_index_in_simdgroup;
    uint first_row=threadgroup_position_in_grid.x*8u
        +simdgroup_index_in_threadgroup*4u;
    uint input_row=threadgroup_position_in_grid.y;
    float sums[4]={{0.0f,0.0f,0.0f,0.0f}};

    for(uint group_index=lane;group_index<groups_per_row;group_index+=32u){{
      float activation[32];
      uint activation_base=input_row*K+group_index*32u;
      for(uint i=0u;i<32u;++i) activation[i]=float(x[activation_base+i]);
      uint block_col=group_index/8u;
      uint group=group_index-block_col*8u;

      for(uint rr=0u;rr<4u;++rr){{
        uint out_row=first_row+rr;
        if(out_row>=N) continue;
        device const uchar* block=packed+(out_row*blocks_per_row+block_col)*110u;
        device const uchar* qs=block+2u+group*8u;
        uint qh=uint(block[66u+group]);
        device const uchar* signs=block+74u+group*4u;
        uint scale_byte=uint(block[106u+group/2u]);
        uint scale=(scale_byte>>(4u*(group&1u)))&15u;
        float d=float(*(reinterpret_cast<device const half*>(block)))*float(1u+2u*scale);
        float acc=0.0f;
        for(uint l=0u;l<4u;++l){{
          uint i0=uint(qs[2u*l])|((qh<<(8u-2u*l))&256u);
          uint i1=uint(qs[2u*l+1u])|((qh<<(7u-2u*l))&256u);
          uint sign_byte=uint(signs[l]);
          for(uint j=0u;j<4u;++j){{
            float v0=grid[i0*4u+j];
            float v1=grid[i1*4u+j];
            acc+=activation[l*8u+j]*((sign_byte&(1u<<j))?-v0:v0);
            acc+=activation[l*8u+j+4u]*((sign_byte&(1u<<(j+4u)))?-v1:v1);
          }}
        }}
        sums[rr]+=d*acc;
      }}
    }}
    for(uint rr=0u;rr<4u;++rr){{
      float total=simd_sum(sums[rr]);
      uint out_row=first_row+rr;
      if(lane==0u && out_row<N) output[input_row*N+out_row]=total;
    }}
    """


@lru_cache(maxsize=None)
def _iq3_s_kernel(in_features: int, out_features: int):
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name=f"mlx_gsq_iq3_s_packed_dot_{in_features}_{out_features}",
        input_names=["x","packed","grid"], output_names=["output"],
        source=_iq3_s_source(in_features, out_features),
    )


@lru_cache(maxsize=1)
def _dummy_tables():
    import mlx.core as mx
    return mx.zeros((1,),dtype=mx.float32),mx.zeros((1,),dtype=mx.uint8)


def quantized_matvec(x, packed, *, qtype: str, out_features: int, in_features: int):
    import mlx.core as mx
    from mlx_gsq.gguf.parser import QUANT_BLOCK_SIZES, GGMLQuantizationType
    type_id=int(GGMLQuantizationType[qtype]); block_elements,block_bytes=QUANT_BLOCK_SIZES[type_id]
    if block_elements!=256 or in_features%256: raise ValueError(f"invalid {qtype} matrix width")
    expected=out_features*(in_features//256)*block_bytes
    if packed.size!=expected: raise ValueError(f"packed size {packed.size} != expected {expected}")
    if x.shape[-1]!=in_features: raise ValueError("input feature size mismatch")
    original=x.shape[:-1]; rows=x.size//in_features; flat=x.reshape((rows,in_features))
    if qtype in _IQ_SOURCES: grid,signs=_tables(qtype)
    elif qtype=="IQ3_S":
        from .iq3_s_decode import _grid
        grid=_grid(); signs=_dummy_tables()[1]
    else: grid,signs=_dummy_tables()
    if qtype=="IQ3_S":
        row_groups=(out_features+7)//8
        out=_iq3_s_kernel(in_features,out_features)(inputs=[flat,packed,grid],grid=(row_groups*64,rows,1),threadgroup=(64,1,1),output_shapes=[(rows,out_features)],output_dtypes=[x.dtype])[0]
    else:
        out=_kernel(qtype,in_features,out_features)(inputs=[flat,packed,grid,signs],grid=(out_features*256,rows,1),threadgroup=(256,1,1),output_shapes=[(rows,out_features)],output_dtypes=[x.dtype])[0]
    return out.reshape((*original,out_features))
