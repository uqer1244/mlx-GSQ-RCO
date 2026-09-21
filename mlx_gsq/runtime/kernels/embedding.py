"""Packed IQ embedding row lookup."""
from __future__ import annotations
from functools import lru_cache
import re

from .iq_lookup_decode import _SOURCES,_tables


def _source():
    body=_SOURCES["IQ2_S"]
    body=re.sub(r"uint\s+index\s*=\s*thread_position_in_grid\.x\s*;","",body,count=1)
    body=re.sub(r"output\s*\[\s*index\s*\]", "output[out_index]",body,count=1)
    return r"""
    uint out_index=thread_position_in_grid.x;
    uint K=shape[0], blocks_per_row=K/256u;
    uint token_pos=out_index/K, feature=out_index-token_pos*K;
    uint token=uint(ids[token_pos]);
    uint index=(token*blocks_per_row+feature/256u)*256u+(feature&255u);
    """+body


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(name="mlx_gsq_iq2_s_embedding",input_names=["ids","packed","grid","ksigns","shape"],output_names=["output"],source=_source())


def iq2_s_embedding(ids,packed,*,vocab_size:int,dimensions:int,dtype=None):
    import mlx.core as mx
    if dimensions%256: raise ValueError("embedding dimensions must be divisible by 256")
    expected=vocab_size*(dimensions//256)*82
    if packed.size!=expected: raise ValueError("packed embedding size mismatch")
    dtype=dtype or mx.bfloat16; flat=ids.reshape(-1); grid,signs=_tables("IQ2_S")
    shape=mx.array([dimensions,vocab_size],dtype=mx.uint32); size=flat.size*dimensions
    out=_kernel()(inputs=[flat,packed,grid,signs,shape],grid=(size,1,1),threadgroup=(256,1,1),output_shapes=[(flat.size,dimensions)],output_dtypes=[dtype])[0]
    return out.reshape((*ids.shape,dimensions))
