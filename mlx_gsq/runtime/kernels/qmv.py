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


def _source(qtype: str) -> str:
    body = _decoder_body(qtype)
    return r"""
    uint tid=thread_index_in_threadgroup;
    uint out_row=threadgroup_position_in_grid.x;
    uint input_row=threadgroup_position_in_grid.y;
    uint K=shape[0], N=shape[1], blocks_per_row=K/256u;
    float sum=0.0f;
    for(uint bc=0;bc<blocks_per_row;++bc){
        uint index=(out_row*blocks_per_row+bc)*256u+tid;
    """ + body + r"""
    }
    threadgroup float partial[256]; partial[tid]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128u;stride>0u;stride>>=1u){
        if(tid<stride) partial[tid]+=partial[tid+stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0u) output[input_row*N+out_row]=partial[0];
    """


@lru_cache(maxsize=None)
def _kernel(qtype: str):
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name=f"mlx_gsq_{qtype.lower()}_qmv",
        input_names=["x","packed","grid","ksigns","shape"],
        output_names=["output"],source=_source(qtype),
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
    shape=mx.array([in_features,out_features],dtype=mx.uint32)
    out=_kernel(qtype)(inputs=[flat,packed,grid,signs,shape],grid=(out_features*256,rows,1),threadgroup=(256,1,1),output_shapes=[(rows,out_features)],output_dtypes=[x.dtype])[0]
    return out.reshape((*original,out_features))
