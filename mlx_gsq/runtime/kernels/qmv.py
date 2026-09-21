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


def _vector_decoder_body(qtype: str) -> str:
    source = _IQ_SOURCES.get(qtype) or _K_SOURCES[qtype]
    source = re.sub(r"uint\s+index\s*=\s*thread_position_in_grid\.x\s*;", "", source, count=1)
    match = list(re.finditer(r"output\s*\[\s*index\s*\]\s*=\s*([^;]+);", source))
    if len(match) != 1:
        raise RuntimeError(f"cannot derive vector QMV decoder body for {qtype}")
    m = match[0]
    replacement = "float decoded_weight = " + m.group(1) + ";\n            sums[rr] += decoded_weight * activation[j];"
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


def _iq2_s_source(in_features: int, out_features: int) -> str:
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
        device const uchar* block=packed+(out_row*blocks_per_row+block_col)*82u;
        device const uchar* qs=block+2u+group*4u;
        device const uchar* signs=block+34u+group*4u;
        uint qh=uint(block[66u+group]);
        uint scales=uint(block[74u+group]);
        float base=float(*(reinterpret_cast<device const half*>(block)))*0.25f;
        float acc0=0.0f,acc1=0.0f;
        for(uint l=0u;l<2u;++l){{
          uint i0=uint(qs[l])|((qh<<(8u-2u*l))&768u);
          uint i1=uint(qs[l+2u])|((qh<<(4u-2u*l))&768u);
          uint s0=uint(signs[l]),s1=uint(signs[l+2u]);
          for(uint j=0u;j<8u;++j){{
            float v0=grid[i0*8u+j],v1=grid[i1*8u+j];
            acc0+=activation[l*8u+j]*((s0&(1u<<j))?-v0:v0);
            acc1+=activation[l*8u+j+16u]*((s1&(1u<<j))?-v1:v1);
          }}
        }}
        sums[rr]+=base*((0.5f+float(scales&15u))*acc0
                       +(0.5f+float(scales>>4u))*acc1);
      }}
    }}
    for(uint rr=0u;rr<4u;++rr){{
      float total=simd_sum(sums[rr]);
      uint out_row=first_row+rr;
      if(lane==0u && out_row<N) output[input_row*N+out_row]=total;
    }}
    """


@lru_cache(maxsize=None)
def _iq2_s_kernel(in_features: int, out_features: int):
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name=f"mlx_gsq_iq2_s_packed_dot_{in_features}_{out_features}",
        input_names=["x","packed","grid"], output_names=["output"],
        source=_iq2_s_source(in_features, out_features),
    )


def _iq3_xxs_source(in_features: int, out_features: int) -> str:
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
        device const uchar* block=packed+(out_row*blocks_per_row+block_col)*98u;
        device const uchar* qs=block+2u+group*8u;
        device const uchar* aux=block+66u+group*4u;
        uint u=uint(aux[0])|(uint(aux[1])<<8u)|(uint(aux[2])<<16u)|(uint(aux[3])<<24u);
        float d=float(*(reinterpret_cast<device const half*>(block)))
            *(0.5f+float(u>>28u))*0.5f;
        float acc=0.0f;
        for(uint l=0u;l<4u;++l){{
          uint sign_byte=uint(ksigns[(u>>(7u*l))&127u]);
          uint i0=uint(qs[2u*l]),i1=uint(qs[2u*l+1u]);
          for(uint j=0u;j<4u;++j){{
            float v0=grid[i0*4u+j],v1=grid[i1*4u+j];
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
def _iq3_xxs_kernel(in_features: int, out_features: int):
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name=f"mlx_gsq_iq3_xxs_packed_dot_{in_features}_{out_features}",
        input_names=["x","packed","grid","ksigns"], output_names=["output"],
        source=_iq3_xxs_source(in_features, out_features),
    )


def _iq4_xs_source(in_features: int, out_features: int) -> str:
    return f"""
    constexpr uint K={in_features}u;
    constexpr uint N={out_features}u;
    constexpr uint blocks_per_row=K/256u;
    constexpr uint groups_per_row=K/32u;
    constexpr int values[16]={{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}};
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
        device const uchar* block=packed+(out_row*blocks_per_row+block_col)*136u;
        uint scales_h=uint(block[2])|(uint(block[3])<<8u);
        uint low_byte=uint(block[4u+group/2u]);
        uint low=(low_byte>>(4u*(group&1u)))&15u;
        uint high=(scales_h>>(2u*group))&3u;
        int scale=int(low|(high<<4u))-32;
        float d=float(*(reinterpret_cast<device const half*>(block)))*float(scale);
        device const uchar* qs=block+8u+group*16u;
        float acc=0.0f;
        for(uint pos=0u;pos<32u;++pos){{
          uint q=(uint(qs[pos&15u])>>(4u*(pos/16u)))&15u;
          acc+=activation[pos]*float(values[q]);
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
def _iq4_xs_kernel(in_features: int, out_features: int):
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name=f"mlx_gsq_iq4_xs_packed_dot_{in_features}_{out_features}",
        input_names=["x","packed"], output_names=["output"],
        source=_iq4_xs_source(in_features, out_features),
    )


_K_PACKED_BODIES = {
    "Q2_K": r"""
        device const uchar* block=packed+(out_row*blocks_per_row+block_col)*84u;
        float d=float(*(reinterpret_cast<device const half*>(block+80u)));
        float dmin=float(*(reinterpret_cast<device const half*>(block+82u)));
        uint chunk=group/4u,plane=group&3u;
        device const uchar* qs=block+16u+chunk*32u;
        float acc=0.0f;
        for(uint pos=0u;pos<32u;++pos){
          uint q=(uint(qs[pos])>>(2u*plane))&3u;
          uint sm=uint(block[group*2u+pos/16u]);
          acc+=activation[pos]*(d*float(sm&15u)*float(q)-dmin*float(sm>>4u));
        }
        sums[rr]+=acc;
    """,
    "Q4_K": r"""
        device const uchar* block=packed+(out_row*blocks_per_row+block_col)*144u;
        device const uchar* scales=block+4u;
        uint sc,mn;
        if(group<4u){
          sc=uint(scales[group])&63u; mn=uint(scales[4u+group])&63u;
        }else{
          uint g=group-4u;
          sc=(uint(scales[8u+g])&15u)|((uint(scales[g])>>2u)&48u);
          mn=(uint(scales[8u+g])>>4u)|((uint(scales[4u+g])>>2u)&48u);
        }
        float d=float(*(reinterpret_cast<device const half*>(block)));
        float dmin=float(*(reinterpret_cast<device const half*>(block+2u)));
        device const uchar* qs=block+16u+(group/2u)*32u;
        uint shift=(group&1u)*4u;
        float acc=0.0f;
        for(uint pos=0u;pos<32u;++pos){
          uint q=(uint(qs[pos])>>shift)&15u;
          acc+=activation[pos]*(d*float(sc)*float(q)-dmin*float(mn));
        }
        sums[rr]+=acc;
    """,
    "Q6_K": r"""
        device const uchar* block=packed+(out_row*blocks_per_row+block_col)*210u;
        uint chunk=group/4u,plane=group&3u;
        float d=float(*(reinterpret_cast<device const half*>(block+208u)));
        float acc=0.0f;
        for(uint pos=0u;pos<32u;++pos){
          uint in_chunk=plane*32u+pos;
          uint ql=(uint(block[chunk*64u+(in_chunk&63u)])>>(4u*(in_chunk/64u)))&15u;
          uint qh=(uint(block[128u+chunk*32u+pos])>>(2u*plane))&3u;
          int q=int(ql|(qh<<4u))-32;
          int scale=int(reinterpret_cast<device const char*>(block+192u)[group*2u+pos/16u]);
          acc+=activation[pos]*d*float(scale*q);
        }
        sums[rr]+=acc;
    """,
}


def _k_packed_source(qtype: str, in_features: int, out_features: int) -> str:
    body = _K_PACKED_BODIES[qtype]
    return f"""
    constexpr uint K={in_features}u;
    constexpr uint N={out_features}u;
    constexpr uint blocks_per_row=K/256u;
    constexpr uint groups_per_row=K/32u;
    uint lane=thread_index_in_simdgroup;
    uint first_row=threadgroup_position_in_grid.x*8u+simdgroup_index_in_threadgroup*4u;
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
    """ + body + r"""
      }
    }
    for(uint rr=0u;rr<4u;++rr){
      float total=simd_sum(sums[rr]);
      uint out_row=first_row+rr;
      if(lane==0u && out_row<N) output[input_row*N+out_row]=total;
    }
    """


@lru_cache(maxsize=None)
def _k_packed_kernel(qtype: str, in_features: int, out_features: int):
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name=f"mlx_gsq_{qtype.lower()}_packed_dot_v2_{in_features}_{out_features}",
        input_names=["x","packed"], output_names=["output"],
        source=_k_packed_source(qtype, in_features, out_features),
    )


_IQ2_PACKED_BODIES = {
    "IQ2_XS": r"""
        device const uchar* block=packed+(out_row*blocks_per_row+block_col)*74u;
        float base=float(*(reinterpret_cast<device const half*>(block)))*0.25f;
        float acc=0.0f;
        for(uint subgroup=0u;subgroup<4u;++subgroup){
          uint o=group*4u+subgroup;
          uint g=o/2u;
          uint q=uint(block[2u+o*2u])|(uint(block[3u+o*2u])<<8u);
          uint sb=uint(block[66u+g/2u]);
          uint scale=(sb>>(4u*(g&1u)))&15u;
          uint sign_byte=uint(ksigns[(q>>9u)&127u]);
          uint qi=q&511u;
          for(uint j=0u;j<8u;++j){
            float v=grid[qi*8u+j];
            acc+=activation[subgroup*8u+j]*base*(0.5f+float(scale))
                *((sign_byte&(1u<<j))?-v:v);
          }
        }
        sums[rr]+=acc;
    """,
    "IQ2_XXS": r"""
        device const uchar* block=packed+(out_row*blocks_per_row+block_col)*66u;
        device const uchar* qs=block+2u+group*8u;
        uint u=uint(qs[4])|(uint(qs[5])<<8u)|(uint(qs[6])<<16u)|(uint(qs[7])<<24u);
        float d=float(*(reinterpret_cast<device const half*>(block)))
            *(0.5f+float(u>>28u))*0.25f;
        float acc=0.0f;
        for(uint l=0u;l<4u;++l){
          uint sign_byte=uint(ksigns[(u>>(7u*l))&127u]);
          uint qi=uint(qs[l]);
          for(uint j=0u;j<8u;++j){
            float v=grid[qi*8u+j];
            acc+=activation[l*8u+j]*((sign_byte&(1u<<j))?-v:v);
          }
        }
        sums[rr]+=d*acc;
    """,
}


def _iq2_packed_source(qtype: str, in_features: int, out_features: int) -> str:
    body = _IQ2_PACKED_BODIES[qtype]
    return f"""
    constexpr uint K={in_features}u;
    constexpr uint N={out_features}u;
    constexpr uint blocks_per_row=K/256u;
    constexpr uint groups_per_row=K/32u;
    uint lane=thread_index_in_simdgroup;
    uint first_row=threadgroup_position_in_grid.x*8u+simdgroup_index_in_threadgroup*4u;
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
    """ + body + r"""
      }
    }
    for(uint rr=0u;rr<4u;++rr){
      float total=simd_sum(sums[rr]);
      uint out_row=first_row+rr;
      if(lane==0u && out_row<N) output[input_row*N+out_row]=total;
    }
    """


@lru_cache(maxsize=None)
def _iq2_packed_kernel(qtype: str, in_features: int, out_features: int):
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name=f"mlx_gsq_{qtype.lower()}_packed_dot_v2_{in_features}_{out_features}",
        input_names=["x","packed","grid","ksigns"], output_names=["output"],
        source=_iq2_packed_source(qtype, in_features, out_features),
    )


def _vector_source(qtype: str, in_features: int, out_features: int) -> str:
    body = _vector_decoder_body(qtype)
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
      for(uint rr=0u;rr<4u;++rr){{
        uint out_row=first_row+rr;
        if(out_row>=N) continue;
        for(uint j=0u;j<32u;++j){{
          uint qelement=group_index*32u+j;
          uint index=out_row*K+qelement;
    """ + body + r"""
        }
      }
    }
    for(uint rr=0u;rr<4u;++rr){
      float total=simd_sum(sums[rr]);
      uint out_row=first_row+rr;
      if(lane==0u && out_row<N) output[input_row*N+out_row]=total;
    }
    """


@lru_cache(maxsize=None)
def _vector_kernel(qtype: str, in_features: int, out_features: int):
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name=f"mlx_gsq_{qtype.lower()}_packed_dot_{in_features}_{out_features}",
        input_names=["x","packed","grid","ksigns"], output_names=["output"],
        source=_vector_source(qtype, in_features, out_features),
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
    elif qtype=="IQ2_S":
        row_groups=(out_features+7)//8
        out=_iq2_s_kernel(in_features,out_features)(inputs=[flat,packed,grid],grid=(row_groups*64,rows,1),threadgroup=(64,1,1),output_shapes=[(rows,out_features)],output_dtypes=[x.dtype])[0]
    elif qtype=="IQ3_XXS":
        row_groups=(out_features+7)//8
        out=_iq3_xxs_kernel(in_features,out_features)(inputs=[flat,packed,grid,signs],grid=(row_groups*64,rows,1),threadgroup=(64,1,1),output_shapes=[(rows,out_features)],output_dtypes=[x.dtype])[0]
    elif qtype=="IQ4_XS":
        row_groups=(out_features+7)//8
        out=_iq4_xs_kernel(in_features,out_features)(inputs=[flat,packed],grid=(row_groups*64,rows,1),threadgroup=(64,1,1),output_shapes=[(rows,out_features)],output_dtypes=[x.dtype])[0]
    elif qtype in _K_PACKED_BODIES:
        row_groups=(out_features+7)//8
        out=_k_packed_kernel(qtype,in_features,out_features)(inputs=[flat,packed],grid=(row_groups*64,rows,1),threadgroup=(64,1,1),output_shapes=[(rows,out_features)],output_dtypes=[x.dtype])[0]
    elif qtype in _IQ2_PACKED_BODIES:
        row_groups=(out_features+7)//8
        out=_iq2_packed_kernel(qtype,in_features,out_features)(inputs=[flat,packed,grid,signs],grid=(row_groups*64,rows,1),threadgroup=(64,1,1),output_shapes=[(rows,out_features)],output_dtypes=[x.dtype])[0]
    else:
        row_groups=(out_features+7)//8
        out=_vector_kernel(qtype,in_features,out_features)(inputs=[flat,packed,grid,signs],grid=(row_groups*64,rows,1),threadgroup=(64,1,1),output_shapes=[(rows,out_features)],output_dtypes=[x.dtype])[0]
    return out.reshape((*original,out_features))
