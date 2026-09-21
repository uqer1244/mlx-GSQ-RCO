"""Metal decoders for lookup-table based IQ formats."""
from __future__ import annotations
from functools import lru_cache

_START=r"""uint index=thread_position_in_grid.x; uint bi=index/256u; uint e=index-bi*256u;"""
_SOURCES={
"IQ3_XXS":_START+r"""
device const uchar* b=packed+bi*98u; uint g=e/32u,p=e-g*32u,l=p/8u,c=p-l*8u;
device const uchar* q=b+2u; device const uchar* s=b+66u; uint u=uint(s[g*4u])|(uint(s[g*4u+1u])<<8u)|(uint(s[g*4u+2u])<<16u)|(uint(s[g*4u+3u])<<24u);
float d=float(*(reinterpret_cast<device const half*>(b)))*(0.5f+float(u>>28u))*0.5f;
uint si=(u>>(7u*l))&127u; bool neg=(uint(ksigns[si])&(1u<<c))!=0u;
uint qi=uint(q[g*8u+l*2u+(c>=4u)]); float v=grid[qi*4u+(c&3u)]; output[index]=neg?-d*v:d*v;""",
"IQ2_XS":_START+r"""
device const uchar* b=packed+bi*74u; uint g=e/16u,p=e-g*16u,sub=p/8u,c=p-sub*8u; uint o=g*2u+sub;
uint q=uint(b[2u+o*2u])|(uint(b[3u+o*2u])<<8u); uint sb=uint(b[66u+g/2u]); uint sc=(sb>>(4u*(g&1u)))&15u;
float d=float(*(reinterpret_cast<device const half*>(b)))*(0.5f+float(sc))*0.25f; uint si=(q>>9u)&127u; bool neg=(uint(ksigns[si])&(1u<<c))!=0u;
float v=grid[(q&511u)*8u+c]; output[index]=neg?-d*v:d*v;""",
"IQ2_S":_START+r"""
device const uchar* b=packed+bi*82u; uint g=e/16u,p=e-g*16u,sub=p/8u,c=p-sub*8u,o=g*2u+sub;
uint hi=(uint(b[66u+o/4u])>>(2u*(o&3u)))&3u; uint qi=uint(b[2u+o])|(hi<<8u); uint sb=uint(b[74u+g/2u]); uint sc=(sb>>(4u*(g&1u)))&15u;
float d=float(*(reinterpret_cast<device const half*>(b)))*(0.5f+float(sc))*0.25f; bool neg=(uint(b[34u+o])&(1u<<c))!=0u;
float v=grid[qi*8u+c]; output[index]=neg?-d*v:d*v;""",
"IQ2_XXS":_START+r"""
device const uchar* b=packed+bi*66u; uint g=e/32u,p=e-g*32u,l=p/8u,c=p-l*8u; device const uchar* q=b+2u+g*8u;
uint u=uint(q[4])|(uint(q[5])<<8u)|(uint(q[6])<<16u)|(uint(q[7])<<24u); float d=float(*(reinterpret_cast<device const half*>(b)))*(0.5f+float(u>>28u))*0.25f;
uint si=(u>>(7u*l))&127u; bool neg=(uint(ksigns[si])&(1u<<c))!=0u; uint qi=uint(q[l]); float v=grid[qi*8u+c]; output[index]=neg?-d*v:d*v;""",
"IQ1_S":_START+r"""
device const uchar* b=packed+bi*50u; uint g=e/32u,p=e-g*32u,l=p/8u,c=p-l*8u; uint h=uint(b[34u+g*2u])|(uint(b[35u+g*2u])<<8u);
uint qi=uint(b[2u+g*4u+l])|(((h>>(3u*l))&7u)<<8u); float d=float(*(reinterpret_cast<device const half*>(b)))*float(2u*((h>>12u)&7u)+1u);
float delta=(h&32768u)?-0.125f:0.125f; output[index]=d*(grid[qi*8u+c]+delta);""",
"IQ1_M":_START+r"""
device const uchar* b=packed+bi*56u; uint g=e/32u,p=e-g*32u,sg=p/16u,r=p-sg*16u,sub=r/8u,c=r-sub*8u,o=g*4u+sg*2u+sub;
uint hb=uint(b[32u+o/2u]); uint hn=(hb>>(4u*(o&1u)))&15u; uint qi=uint(b[o])|((hn&7u)<<8u); float delta=(hn&8u)?-0.125f:0.125f;
device const uchar* sp=b+48u; uint w0=uint(sp[0])|(uint(sp[1])<<8u),w1=uint(sp[2])|(uint(sp[3])<<8u),w2=uint(sp[4])|(uint(sp[5])<<8u),w3=uint(sp[6])|(uint(sp[7])<<8u);
uint dbits=((w0&61440u)>>12u)|((w1&61440u)>>8u)|((w2&61440u)>>4u)|(w3&61440u); float d=float(as_type<half>(ushort(dbits)));
uint sci=g*2u+sg, wi=sci/4u, sh=3u*(sci&3u); uint w=wi==0u?w0:(wi==1u?w1:(wi==2u?w2:w3)); uint sc=(w>>sh)&7u;
output[index]=d*float(2u*sc+1u)*(grid[qi*8u+c]+delta);""",
}
_BYTES={"IQ1_M":56,"IQ1_S":50,"IQ2_XXS":66,"IQ2_XS":74,"IQ2_S":82,"IQ3_XXS":98}

@lru_cache(maxsize=None)
def _kernel(q):
 import mlx.core as mx
 return mx.fast.metal_kernel(name=f"mlx_gsq_decode_{q.lower()}",input_names=["packed","grid","ksigns"],output_names=["output"],source=_SOURCES[q])

@lru_cache(maxsize=None)
def _tables(q):
 import mlx.core as mx
 from gguf import quants
 cls=getattr(quants,q); cls.init_grid()
 grid=mx.array(cls.grid.reshape(-1),dtype=mx.float32)
 signs=mx.array(list(quants.IQ2_XXS.ksigns),dtype=mx.uint8)
 return grid,signs

def decode_lookup_quant(packed,qtype):
 import mlx.core as mx
 if qtype not in _SOURCES: raise KeyError(qtype)
 if packed.dtype!=mx.uint8: raise TypeError("packed input must be uint8")
 bs=_BYTES[qtype]
 if packed.size%bs: raise ValueError(f"{qtype} payload is not block aligned")
 size=packed.size//bs*256; grid,signs=_tables(qtype)
 return _kernel(qtype)(inputs=[packed,grid,signs],grid=(size,1,1),threadgroup=(256,1,1),output_shapes=[(size,)],output_dtypes=[mx.float32])[0]
