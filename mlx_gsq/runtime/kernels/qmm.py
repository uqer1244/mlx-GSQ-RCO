"""Tiled fused QMM kernels for prefill (multi-row inputs).

Design
------
A threadgroup computes a tile of ``M_TILE x N_TILE`` outputs over one
``K_TILE = 256`` element (i.e. one quantization block column) at a time:

1. The 256-thread group cooperatively decodes the ``N_TILE`` packed weight
   blocks of this column *once* into a threadgroup-memory half tile.
   This amortizes the per-element GGUF unpack cost over all ``M_TILE``
   input rows, which is the main cost of the per-row QMV prefill path.
2. The ``M_TILE`` activation rows of the column are loaded into a second
   threadgroup-memory tile with 16-byte vector loads.
3. Each thread accumulates 2 output elements (2 N rows x 1 M row) over the
   256-wide column by dotting threadgroup-memory tiles.

The per-element decode expressions are the validated ones used by the
decoder kernels; only the addressing (tile row / block column) changes.

Dispatch: ``quantized_matvec`` routes inputs with ``rows >= QMM_MIN_ROWS``
here and keeps the per-row QMV kernel for very small row counts.
"""
from __future__ import annotations
from functools import lru_cache

QMM_MIN_ROWS = 8
M_TILE = 16
N_TILE = 32
K_TILE = 256
_WP = K_TILE + 4  # padded row stride (520 B): 8 B-aligned for half4 access, odd word offset (130) avoids bank conflicts

# Per-format decode body.  Context supplied by the template:
#   b   : device const uchar* pointing at the packed block
#   g   : group within the 256-element block (0..7)
#   p   : position within the group (0..31)
#   grid / ksigns : lookup tables (unused by some formats)
# The body must assign the decoded element to ``value``.
_DECODE_BODIES = {
    "IQ1_M": (56, r"""
        uint sg=p/16u, r=p-sg*16u, sub=r/8u, c=r-sub*8u, o=g*4u+sg*2u+sub;
        uint hb=uint(b[32u+o/2u]); uint hn=(hb>>(4u*(o&1u)))&15u;
        uint qi=uint(b[o])|((hn&7u)<<8u); float delta=(hn&8u)?-0.125f:0.125f;
        device const uchar* sp=b+48u;
        uint w0=uint(sp[0])|(uint(sp[1])<<8u), w1=uint(sp[2])|(uint(sp[3])<<8u);
        uint w2=uint(sp[4])|(uint(sp[5])<<8u), w3=uint(sp[6])|(uint(sp[7])<<8u);
        uint dbits=((w0&61440u)>>12u)|((w1&61440u)>>8u)|((w2&61440u)>>4u)|(w3&61440u);
        float d=float(as_type<half>(ushort(dbits)));
        uint sci=g*2u+sg, wi=sci/4u, sh=3u*(sci&3u);
        uint ww=wi==0u?w0:(wi==1u?w1:(wi==2u?w2:w3)); uint sc=(ww>>sh)&7u;
        value=d*float(2u*sc+1u)*(grid[qi*8u+c]+delta);"""),
    "IQ1_S": (50, r"""
        uint l=p/8u, c=p-l*8u;
        uint h=uint(b[34u+g*2u])|(uint(b[35u+g*2u])<<8u);
        uint qi=uint(b[2u+g*4u+l])|(((h>>(3u*l))&7u)<<8u);
        float d=float(*(reinterpret_cast<device const half*>(b)))*float(2u*((h>>12u)&7u)+1u);
        float delta=(h&32768u)?-0.125f:0.125f;
        value=d*(grid[qi*8u+c]+delta);"""),
    "IQ2_XXS": (66, r"""
        uint l=p/8u, c=p-l*8u;
        device const uchar* q=b+2u+g*8u;
        uint u=uint(q[4])|(uint(q[5])<<8u)|(uint(q[6])<<16u)|(uint(q[7])<<24u);
        float d=float(*(reinterpret_cast<device const half*>(b)))*(0.5f+float(u>>28u))*0.25f;
        uint si=(u>>(7u*l))&127u; bool neg=(uint(ksigns[si])&(1u<<c))!=0u;
        uint qi=uint(q[l]); float v=grid[qi*8u+c];
        value=neg?-d*v:d*v;"""),
    "IQ2_XS": (74, r"""
        uint e=g*32u+p, gg=e/16u, pp=e-gg*16u, sub=pp/8u, c=pp-sub*8u, o=gg*2u+sub;
        uint q=uint(b[2u+o*2u])|(uint(b[3u+o*2u])<<8u);
        uint sb=uint(b[66u+gg/2u]); uint sc=(sb>>(4u*(gg&1u)))&15u;
        float d=float(*(reinterpret_cast<device const half*>(b)))*(0.5f+float(sc))*0.25f;
        uint si=(q>>9u)&127u; bool neg=(uint(ksigns[si])&(1u<<c))!=0u;
        float v=grid[(q&511u)*8u+c]; value=neg?-d*v:d*v;"""),
    "IQ2_S": (82, r"""
        uint e=g*32u+p, gg=e/16u, pp=e-gg*16u, sub=pp/8u, c=pp-sub*8u, o=gg*2u+sub;
        uint hi=(uint(b[66u+o/4u])>>(2u*(o&3u)))&3u;
        uint qi=uint(b[2u+o])|(hi<<8u);
        uint sb=uint(b[74u+gg/2u]); uint sc=(sb>>(4u*(gg&1u)))&15u;
        float d=float(*(reinterpret_cast<device const half*>(b)))*(0.5f+float(sc))*0.25f;
        bool neg=(uint(b[34u+o])&(1u<<c))!=0u;
        float v=grid[qi*8u+c]; value=neg?-d*v:d*v;"""),
    "IQ3_XXS": (98, r"""
        uint l=p/8u, c=p-l*8u;
        device const uchar* q=b+2u; device const uchar* s=b+66u;
        uint u=uint(s[g*4u])|(uint(s[g*4u+1u])<<8u)|(uint(s[g*4u+2u])<<16u)|(uint(s[g*4u+3u])<<24u);
        float d=float(*(reinterpret_cast<device const half*>(b)))*(0.5f+float(u>>28u))*0.5f;
        uint si=(u>>(7u*l))&127u; bool neg=(uint(ksigns[si])&(1u<<c))!=0u;
        uint qi=uint(q[g*8u+l*2u+(c>=4u)]); float v=grid[qi*4u+(c&3u)];
        value=neg?-d*v:d*v;"""),
    "IQ3_S": (110, r"""
        uint l=p/8u, c=p-l*8u;
        float d=float(*(reinterpret_cast<device const half*>(b)));
        uchar scale_byte=b[106u+(g>>1u)];
        uint snib=(g&1u)?(uint(scale_byte)>>4u):(uint(scale_byte)&15u);
        float db=d*float(1u+2u*snib);
        uint qh=uint(b[66u+g]);
        uint gi=(c<4u)
            ? (uint(b[2u+g*8u+l*2u])|((uint(qh)<<(8u-2u*l))&256u))
            : (uint(b[2u+g*8u+l*2u+1u])|((uint(qh)<<(7u-2u*l))&256u));
        bool neg=(uint(b[74u+g*4u+l])&(1u<<c))!=0u;
        float mag=grid[gi*4u+(c&3u)];
        value=neg?-db*mag:db*mag;"""),
    "IQ4_XS": (136, r"""
        uint scales_h=uint(b[2u])|(uint(b[3u])<<8u);
        uint low_byte=uint(b[4u+g/2u]);
        uint low=(low_byte>>(4u*(g&1u)))&15u;
        uint high=(scales_h>>(2u*g))&3u;
        int scale=int(low|(high<<4u))-32;
        uint q=(uint(b[8u+g*16u+(p&15u)])>>(4u*(p/16u)))&15u;
        float d=float(*(reinterpret_cast<device const half*>(b)));
        value=d*float(scale)*float(values[q]);"""),
    "Q2_K": (84, r"""
        uint e=g*32u+p, chunk=e/128u, in_chunk=e-chunk*128u;
        uint plane=in_chunk/32u, pos=in_chunk-plane*32u;
        uint q=(uint(b[16u+chunk*32u+pos])>>(2u*plane))&3u;
        uint g16=e/16u; uint sm=uint(b[g16]);
        float d=float(*(reinterpret_cast<device const half*>(b+80u)));
        float dmin=float(*(reinterpret_cast<device const half*>(b+82u)));
        value=d*float(sm&15u)*float(q)-dmin*float(sm>>4u);"""),
    "Q4_K": (144, r"""
        device const uchar* s=b+4u;
        uint qbyte=uint(b[16u+(g/2u)*32u+p]);
        uint q=(qbyte>>(4u*(g&1u)))&15u;
        uint sc,mn;
        if(g<4u){ sc=uint(s[g])&63u; mn=uint(s[4u+g])&63u; }
        else{
          uint gg=g-4u;
          sc=(uint(s[8u+gg])&15u)|((uint(s[gg])>>2u)&48u);
          mn=(uint(s[8u+gg])>>4u)|((uint(s[4u+gg])>>2u)&48u);
        }
        float d=float(*(reinterpret_cast<device const half*>(b)));
        float dmin=float(*(reinterpret_cast<device const half*>(b+2u)));
        value=d*float(sc)*float(q)-dmin*float(mn);"""),
    "Q6_K": (210, r"""
        uint e=g*32u+p, chunk=e/128u, in_chunk=e-chunk*128u;
        uint ql_half=in_chunk/64u, ql_pos=in_chunk-ql_half*64u;
        uint ql=(uint(b[chunk*64u+ql_pos])>>(4u*ql_half))&15u;
        uint qh_plane=in_chunk/32u, qh_pos=in_chunk-qh_plane*32u;
        uint qh=(uint(b[128u+chunk*32u+qh_pos])>>(2u*qh_plane))&3u;
        int q=int(ql|(qh<<4u))-32;
        int scale=int(reinterpret_cast<device const char*>(b+192u)[e/16u]);
        float d=float(*(reinterpret_cast<device const half*>(b+208u)));
        value=d*float(scale)*float(q);"""),
}

# Kernel top-level extra declarations per qtype.
_TOP_DECLS = {
    "IQ4_XS": "constexpr int values[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};",
}


def _x_load(is_half: bool) -> str:
    # Each thread loads exactly 8 elements (16 B for fp16, 32 B for fp32);
    # koff = (h % 32) * 8 gives the 8-half slot window of the thread.
    if is_half:
        return r"""
            half4 xa=*reinterpret_cast<const device half4*>(&x[xb]);
            half4 xb2=*reinterpret_cast<const device half4*>(&x[xb+4u]);
            half4* xd=reinterpret_cast<half4*>(&xtile[ml2*WP+koff]);
            xd[0u]=xa;
            xd[1u]=xb2;"""
    return r"""
            float4 a=*reinterpret_cast<const device float4*>(&x[xb]);
            float4 bf=*reinterpret_cast<const device float4*>(&x[xb+4u]);
            half4* xd=reinterpret_cast<half4*>(&xtile[ml2*WP+koff]);
            xd[0u]=half4(half(a.x),half(a.y),half(a.z),half(a.w));
            xd[1u]=half4(half(bf.x),half(bf.y),half(bf.z),half(bf.w));"""


_DECODE_TPL = """      {{
@DECLS@        uint nl=@@NL@@;
        uint g=@@G@@;
        uint ng=n0+nl;
        if(ng<N){{
          device const uchar* b=packed+@@BPTR@@;
          for(uint q=0u;q<@@QMAX@@;++q){{
            half lv[4];
            for(uint rr=0u;rr<4u;++rr){{
              uint p=@@P@@;
              float value=0.0f;
@BODY@
              lv[rr]=half(value);
            }}
            *reinterpret_cast<half4*>(&wtile[nl*WP+@@OFF@@])=half4(lv[0u],lv[1u],lv[2u],lv[3u]);
          }}
        }}else{{
          half z2=(half)0;
          half4 zq;
          zq.x=zq.y=zq.z=zq.w=z2;
          half4* zd2=reinterpret_cast<half4*>(&wtile[nl*WP+@@OFF0@@]);
          for(uint q=0u;q<@@QMAX@@;++q) zd2[q]=zq;
        }}
      }}"""

_GEMM_FMA_TPL = """      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(m_ok){{
        const half4* xr0=reinterpret_cast<const half4*>(&xtile[ml*WP]);
        const half4* wr0=reinterpret_cast<const half4*>(&wtile[nl0*WP]);
        const half4* wr1=reinterpret_cast<const half4*>(&wtile[(nl0+1u)*WP]);
        half pa0=(half)0, pa1=(half)0;
        for(uint c=0u;c<8u;++c){{
          const half4* xr=xr0+c*@@CS@@u;
          const half4* w0=wr0+c*@@CS@@u;
          const half4* w1=wr1+c*@@CS@@u;
          for(uint j4=0u;j4<@@J4@@u;++j4){{
            pa0+=dot(xr[j4],w0[j4]);
            if(n0+nl0+1u<N) pa1+=dot(xr[j4],w1[j4]);
          }}
        }}
        acc0+=float(pa0);
        if(n0+nl0+1u<N) acc1+=float(pa1);
      }}"""

_GEMM_F32_TPL = """      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(m_ok){{
        const half4* xr0=reinterpret_cast<const half4*>(&xtile[ml*WP]);
        const half4* wr0=reinterpret_cast<const half4*>(&wtile[nl0*WP]);
        const half4* wr1=reinterpret_cast<const half4*>(&wtile[(nl0+1u)*WP]);
        for(uint c=0u;c<8u;++c){{
          const half4* xr=xr0+c*@@CS@@u;
          const half4* w0=wr0+c*@@CS@@u;
          const half4* w1=wr1+c*@@CS@@u;
          for(uint j4=0u;j4<@@J4@@u;++j4){{
            float4 xh=float4(xr[j4]);
            acc0+=dot(xh,float4(w0[j4]));
            if(n0+nl0+1u<N) acc1+=dot(xh,float4(w1[j4]));
          }}
        }}
      }}"""

# Tile configurations. KT=256 is the production layout (TSM 24.9KB, one
# threadgroup per GPU core). KT=128 halves the TSM footprint (12.7KB) so two
# threadgroups fit per core and their decode/GEMM phases can overlap across
# tiles; global x traffic (rows*N*K*2B/NT) and total FLOPs are unchanged.
_KT_CFGS = {
    256: dict(
        wp=260, reps=2, xd=32, cs=8, j4=8,
        decls="", nl="tid/8u", g="tid%8u",
        bptr="(ng*K_BLOCKS+bc)*{bb}u", qmax="8u",
        p="q*4u+rr", off="g*32u+q*4u", off0="g*32u"),
    128: dict(
        wp=132, reps=1, xd=16, cs=4, j4=4,
        decls=("uint blk=tid/2u;\n"
               "        uint sub=tid%2u;\n"
               "        uint gl=blk%4u;\n"
               "        "),
        nl="blk/4u", g="(bc&1u)*4u+gl",
        bptr="(ng*K_BLOCKS+bc/2u)*{bb}u", qmax="4u",
        p="sub*16u+q*4u+rr", off="gl*32u+sub*16u+q*4u", off0="gl*32u+sub*16u"),
}


def _source(qtype: str, in_features: int, out_features: int, is_half: bool,
            half_fma: bool = False, kt: int = 256) -> str:
    if kt not in _KT_CFGS:
        raise ValueError(f"unsupported kt {kt}")
    cfg = _KT_CFGS[kt]
    block_bytes, body = _DECODE_BODIES[qtype]
    top = _TOP_DECLS.get(qtype, "")
    x_load = _x_load(is_half)
    decode = (_DECODE_TPL
              .replace("@DECLS@", cfg["decls"])
              .replace("@@NL@@", cfg["nl"])
              .replace("@@G@@", cfg["g"])
              .replace("@@BPTR@@", cfg["bptr"].format(bb=block_bytes))
              .replace("@@QMAX@@", cfg["qmax"])
              .replace("@@P@@", cfg["p"])
              .replace("@@OFF0@@", cfg["off0"]).replace("@@OFF@@", cfg["off"])
              .replace("@BODY@", body))
    gemm = (_GEMM_FMA_TPL if half_fma else _GEMM_F32_TPL)
    gemm = gemm.replace("@@CS@@", str(cfg["cs"])).replace("@@J4@@", str(cfg["j4"]))
    return f"""
    constexpr uint K={in_features}u;
    constexpr uint N={out_features}u;
    constexpr uint K_BLOCKS=K/256u;
    constexpr uint MT={M_TILE}u;
    constexpr uint NT={N_TILE}u;
    constexpr uint KT={kt}u;
    constexpr uint WP={cfg['wp']}u;
{top}
    struct h8 {{ half s,t,u,v,w,x,y,z; }};
    uint M=meta[0];
    uint row_tiles=meta[1];
    uint tid=thread_index_in_threadgroup;
    // MLX `grid` is the total thread count, so the threadgroup grid is 1D;
    // map the linear threadgroup index back to (tile_n, tile_m).
    uint tile_idx=threadgroup_position_in_grid.x;
    uint tile_n=tile_idx/row_tiles;
    uint tile_m=tile_idx-row_tiles*tile_n;
    uint n0=tile_n*NT;
    uint m0=tile_m*MT;
    threadgroup half wtile[NT*WP];
    threadgroup half xtile[MT*WP];
    uint ml=tid%MT;
    uint cg=tid/MT;
    uint nl0=cg*2u;
    uint m_global=m0+ml;
    bool m_ok=m_global<M;
    float acc0=0.0f, acc1=0.0f;
    for(uint bc=0u;bc<K/KT;++bc){{
      for(uint rep=0u;rep<{cfg['reps']}u;++rep){{
        uint h=tid+256u*rep;
        uint ml2=h/{cfg['xd']}u;
        uint koff=(h%{cfg['xd']}u)*8u;
        uint xrow=m0+ml2;
        if(xrow<M){{
          uint xb=xrow*K+bc*KT+koff;
{x_load}
        }}else{{
          half4 z; z.x=z.y=z.z=z.w=(half)0;
          half4* zd=reinterpret_cast<half4*>(&xtile[ml2*WP+koff]);
          zd[0u]=z; zd[1u]=z;
        }}
      }}
{decode}
      threadgroup_barrier(mem_flags::mem_threadgroup);
{gemm}
      // Second barrier: every thread must finish reading the tiles before
      // any thread may overwrite them with the next block-column's data.
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if(m_ok){{
      if(n0+nl0<N) output[m_global*N+n0+nl0]=acc0;
      if(n0+nl0+1u<N) output[m_global*N+n0+nl0+1u]=acc1;
    }}
    """

def _source_outer(qtype: str, in_features: int, out_features: int, is_half: bool) -> str:
    """Outer-product variant (EXPERIMENTAL - do not route to).

    Each thread owns (n, k-slice of 32) and decodes its quantization block
    once into 8 float4 registers. The GEMM then loops over the 16 m rows,
    streaming only x from threadgroup memory. k-split partials are combined
    through a small threadgroup reduction buffer (2 barriers per block
    column, same as the TSM-tile variant).

    Measured 0.67-0.83x SLOWER than :func:`quantized_matmul` on M3 Pro:
    each x element is read by all NT=32 n-threads with a 260-half (65-
    word) row stride, which produces ~8-way TSM bank conflicts. Kept as a
    reference for future GPUs / 2D register-tiling redesigns.
    """
    block_bytes, body = _DECODE_BODIES[qtype]
    top = _TOP_DECLS.get(qtype, "")
    x_load = _x_load(is_half)
    return f"""
    constexpr uint K={in_features}u;
    constexpr uint N={out_features}u;
    constexpr uint K_BLOCKS=K/256u;
    constexpr uint MT={M_TILE}u;
    constexpr uint NT={N_TILE}u;
    constexpr uint KT={K_TILE}u;
    constexpr uint WP={_WP}u;
{top}
    uint M=meta[0];
    uint row_tiles=meta[1];
    uint tid=thread_index_in_threadgroup;
    uint tile_idx=threadgroup_position_in_grid.x;
    uint tile_n=tile_idx/row_tiles;
    uint tile_m=tile_idx-row_tiles*tile_n;
    uint n0=tile_n*NT;
    uint m0=tile_m*MT;
    threadgroup half xtile[MT*WP];
    threadgroup float red[256*16];
    uint nl=tid/8u;
    uint ks=tid%8u;
    uint ng=n0+nl;
    bool n_ok=ng<N;
    uint rbase=ng*K_BLOCKS;
    float acc[16];
    for(uint m=0u;m<16u;++m) acc[m]=0.0f;
    for(uint bc=0u;bc<K_BLOCKS;++bc){{
      for(uint rep=0u;rep<2u;++rep){{
        uint h=tid+256u*rep;
        uint ml2=h/32u;
        uint koff=(h%32u)*8u;
        uint xrow=m0+ml2;
        if(xrow<M){{
          uint xb=xrow*K+bc*KT+koff;
{x_load}
        }}else{{
          half4 z; z.x=z.y=z.z=z.w=(half)0;
          half4* zd=reinterpret_cast<half4*>(&xtile[ml2*WP+koff]);
          zd[0u]=z; zd[1u]=z;
        }}
      }}
      float4 wv4[8];
      float part[16];
      if(n_ok){{
        device const uchar* b=packed+(rbase+bc)*{block_bytes}u;
        uint g=ks;
        float wv[32];
        for(uint p=0u;p<32u;++p){{
          float value=0.0f;
{body}
          wv[p]=value;
        }}
        wv4[0u]=float4(wv[0u],wv[1u],wv[2u],wv[3u]);
        wv4[1u]=float4(wv[4u],wv[5u],wv[6u],wv[7u]);
        wv4[2u]=float4(wv[8u],wv[9u],wv[10u],wv[11u]);
        wv4[3u]=float4(wv[12u],wv[13u],wv[14u],wv[15u]);
        wv4[4u]=float4(wv[16u],wv[17u],wv[18u],wv[19u]);
        wv4[5u]=float4(wv[20u],wv[21u],wv[22u],wv[23u]);
        wv4[6u]=float4(wv[24u],wv[25u],wv[26u],wv[27u]);
        wv4[7u]=float4(wv[28u],wv[29u],wv[30u],wv[31u]);
      }}else{{
        for(uint q=0u;q<8u;++q) wv4[q]=float4(0.0f);
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      {{
        for(uint m=0u;m<16u;++m){{
          const half4* xr=reinterpret_cast<const half4*>(&xtile[m*WP+ks*32u]);
          float s=0.0f;
          for(uint q=0u;q<8u;++q) s+=dot(float4(xr[q]),wv4[q]);
          part[m]=s;
        }}
      }}
      {{
        float4* rw=reinterpret_cast<float4*>(&red[tid*16u]);
        for(uint m4=0u;m4<4u;++m4)
          rw[m4]=float4(part[m4*4u],part[m4*4u+1u],part[m4*4u+2u],part[m4*4u+3u]);
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      {{
        float4 s4[4];
        for(uint m4=0u;m4<4u;++m4) s4[m4]=float4(0.0f);
        for(uint k2=0u;k2<8u;++k2){{
          const float4* r4=reinterpret_cast<const float4*>(&red[(nl*8u+k2)*16u]);
          for(uint m4=0u;m4<4u;++m4) s4[m4]+=r4[m4];
        }}
        for(uint m4=0u;m4<4u;++m4){{
          acc[m4*4u]+=s4[m4].x;
          acc[m4*4u+1u]+=s4[m4].y;
          acc[m4*4u+2u]+=s4[m4].z;
          acc[m4*4u+3u]+=s4[m4].w;
        }}
      }}
    }}
    if(ks==0u && n_ok){{
      for(uint m=0u;m<16u;++m){{
        uint m_global=m0+m;
        if(m_global<M) output[m_global*N+ng]=acc[m];
      }}
    }}
    """


@lru_cache(maxsize=None)
def _qmm_outer_kernel(qtype: str, in_features: int, out_features: int, is_half: bool):
    import mlx.core as mx
    return mx.fast.metal_kernel(
        name=f"mlx_gsq_{qtype.lower()}_{in_features}_{out_features}_qmm_op_{'f16' if is_half else 'f32'}",
        input_names=["x", "packed", "grid", "ksigns", "meta"],
        output_names=["output"],
        source=_source_outer(qtype, in_features, out_features, is_half),
    )


@lru_cache(maxsize=None)
def _qmm_kernel(qtype: str, in_features: int, out_features: int, is_half: bool, half_fma: bool = False, kt: int = 256):
    import mlx.core as mx
    tag = ("hf" if half_fma else "fma") + (f"_{kt}" if kt != 256 else "")
    return mx.fast.metal_kernel(
        name=f"mlx_gsq_{qtype.lower()}_{in_features}_{out_features}_qmm_{tag}_{'f16' if is_half else 'f32'}",
        input_names=["x", "packed", "grid", "ksigns", "meta"],
        output_names=["output"],
        source=_source(qtype, in_features, out_features, is_half, half_fma=half_fma, kt=kt),
    )


@lru_cache(maxsize=None)
def _tables(qtype: str):
    from .iq_lookup_decode import _SOURCES as _iq_types, _tables as _iq_tables
    if qtype in _iq_types:
        return _iq_tables(qtype)
    if qtype == "IQ3_S":
        from .iq3_s_decode import _grid
        from .qmv import _dummy_tables
        return _grid(), _dummy_tables()[1]
    from .qmv import _dummy_tables
    return _dummy_tables()


def quantized_matmul_outer(x, packed, *, qtype: str, out_features: int, in_features: int):
    """Outer-product QMM variant (decoded weights kept in fp32 registers).

    Same contract as :func:`quantized_matmul`.
    """
    import mlx.core as mx
    from mlx_gsq.gguf.parser import QUANT_BLOCK_SIZES, GGMLQuantizationType

    if qtype not in _DECODE_BODIES:
        raise NotImplementedError(f"no tiled QMM kernel for {qtype}")
    type_id = int(GGMLQuantizationType[qtype])
    block_elements, block_bytes = QUANT_BLOCK_SIZES[type_id]
    if block_elements != 256 or in_features % 256:
        raise ValueError(f"invalid {qtype} matrix width")
    expected = out_features * (in_features // 256) * block_bytes
    if packed.size != expected:
        raise ValueError(f"packed size {packed.size} != expected {expected}")
    if x.shape[-1] != in_features:
        raise ValueError("input feature size mismatch")
    original = x.shape[:-1]
    rows = x.size // in_features
    flat = x.reshape((rows, in_features))
    grid, signs = _tables(qtype)
    row_tiles = (rows + M_TILE - 1) // M_TILE
    col_tiles = (out_features + N_TILE - 1) // N_TILE
    meta = mx.array([rows, row_tiles], dtype=mx.uint32)
    out = _qmm_outer_kernel(qtype, in_features, out_features, x.dtype == mx.float16)(
        inputs=[flat, packed, grid, signs, meta],
        grid=(col_tiles * row_tiles * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, out_features)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape((*original, out_features))


def quantized_matmul_hf(x, packed, *, qtype: str, out_features: int, in_features: int):
    """QMM variant with half-precision FMA inside the GEMM loop (EXPERIMENTAL).

    Same contract as :func:`quantized_matmul`; per-block-column partials are
    accumulated in half and folded into the fp32 accumulator after each 256-
    wide column, which removes all half->float conversions from the inner
    loop. Measured ~1.01-1.03x at rows=128 on M3 Pro (fp16 FMA runs at the
    same rate as fp32 there) and 1.3x at rows=16 (latency regime); precision
    cost ~1.4e-3 relative at K=5120. Not routed by default.
    """
    import mlx.core as mx
    from mlx_gsq.gguf.parser import QUANT_BLOCK_SIZES, GGMLQuantizationType

    if qtype not in _DECODE_BODIES:
        raise NotImplementedError(f"no tiled QMM kernel for {qtype}")
    type_id = int(GGMLQuantizationType[qtype])
    block_elements, block_bytes = QUANT_BLOCK_SIZES[type_id]
    if block_elements != 256 or in_features % 256:
        raise ValueError(f"invalid {qtype} matrix width")
    expected = out_features * (in_features // 256) * block_bytes
    if packed.size != expected:
        raise ValueError(f"packed size {packed.size} != expected {expected}")
    if x.shape[-1] != in_features:
        raise ValueError("input feature size mismatch")
    original = x.shape[:-1]
    rows = x.size // in_features
    flat = x.reshape((rows, in_features))
    grid, signs = _tables(qtype)
    row_tiles = (rows + M_TILE - 1) // M_TILE
    col_tiles = (out_features + N_TILE - 1) // N_TILE
    meta = mx.array([rows, row_tiles], dtype=mx.uint32)
    out = _qmm_kernel(qtype, in_features, out_features, x.dtype == mx.float16, half_fma=True)(
        inputs=[flat, packed, grid, signs, meta],
        grid=(col_tiles * row_tiles * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, out_features)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape((*original, out_features))


def quantized_matmul(x, packed, *, qtype: str, out_features: int, in_features: int, kt: int = 256):
    """Fused quantized matrix-matrix for prefill (multi-row inputs).

    Same validation contract as ``qmv.quantized_matvec``. ``kt`` selects the
    K-tile width (256 production; 128 halves the TSM footprint so two
    threadgroups fit per core and their phases overlap).
    """
    import mlx.core as mx
    from mlx_gsq.gguf.parser import QUANT_BLOCK_SIZES, GGMLQuantizationType

    if qtype not in _DECODE_BODIES:
        raise NotImplementedError(f"no tiled QMM kernel for {qtype}")
    type_id = int(GGMLQuantizationType[qtype])
    block_elements, block_bytes = QUANT_BLOCK_SIZES[type_id]
    if block_elements != 256 or in_features % 256:
        raise ValueError(f"invalid {qtype} matrix width")
    expected = out_features * (in_features // 256) * block_bytes
    if packed.size != expected:
        raise ValueError(f"packed size {packed.size} != expected {expected}")
    if x.shape[-1] != in_features:
        raise ValueError("input feature size mismatch")
    original = x.shape[:-1]
    rows = x.size // in_features
    flat = x.reshape((rows, in_features))
    grid, signs = _tables(qtype)
    row_tiles = (rows + M_TILE - 1) // M_TILE
    col_tiles = (out_features + N_TILE - 1) // N_TILE
    meta = mx.array([rows, row_tiles], dtype=mx.uint32)
    out = _qmm_kernel(qtype, in_features, out_features, x.dtype == mx.float16, kt=kt)(
        inputs=[flat, packed, grid, signs, meta],
        grid=(col_tiles * row_tiles * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, out_features)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape((*original, out_features))
