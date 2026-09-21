"""Attach packed GSQ weights to mlx-lm's Qwen3.5 implementation."""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from mlx_gsq.nn import GSQEmbedding,GSQLinear


def _args():
    from mlx_lm.models.qwen3_5 import TextModelArgs
    return TextModelArgs(
        model_type="qwen3_5",
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        vocab_size=248320,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        linear_num_value_heads=48,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        tie_word_embeddings=False,
        attention_bias=False,
        rope_parameters={
            "type":"default","mrope_section":[11,11,10],
            "rope_theta":10_000_000.0,"partial_rotary_factor":0.25,
        },
    )


def _plain(store,name):
    return store.materialize_plain(name)


def _restore_a_log(ssm_a):
    """Convert GGUF's ``-exp(A_log)`` coefficient back to ``A_log``."""
    return mx.log(-ssm_a)


def _gguf_gated_delta_call(self,inputs,mask=None,cache=None):
    """GDN forward with llama.cpp's cyclic Q/K-to-value-head routing."""
    import mlx.nn as nn
    from mlx_lm.models.gated_delta import gated_delta_update

    B,S,_=inputs.shape
    if self.sharding_group is not None:
        from mlx.nn.layers.distributed import sum_gradients
        inputs=sum_gradients(self.sharding_group)(inputs)
    qkv=self.in_proj_qkv(inputs)
    z=self.in_proj_z(inputs).reshape(B,S,self.num_v_heads,self.head_v_dim)
    b=self.in_proj_b(inputs)
    a=self.in_proj_a(inputs)
    conv_state=(cache[0] if cache is not None and cache[0] is not None else
                mx.zeros((B,self.conv_kernel_size-1,self.conv_dim),dtype=inputs.dtype))
    if mask is not None:
        qkv=mx.where(mask[...,None],qkv,0)
    conv_input=mx.concatenate([conv_state,qkv],axis=1)
    if cache is not None:
        n_keep=self.conv_kernel_size-1
        if cache.lengths is not None:
            ends=mx.clip(cache.lengths,0,S)
            positions=(ends[:,None]+mx.arange(n_keep))[...,None]
            cache[0]=mx.take_along_axis(conv_input,positions,axis=1)
        else:
            cache[0]=mx.contiguous(conv_input[:,-n_keep:,:])
    conv_out=nn.silu(self.conv1d(conv_input))
    q,k,v=[
        t.reshape(B,S,h,d)
        for t,h,d in zip(
            mx.split(conv_out,[self.key_dim,2*self.key_dim],-1),
            [self.num_k_heads,self.num_k_heads,self.num_v_heads],
            [self.head_k_dim,self.head_k_dim,self.head_v_dim],
        )
    ]
    state=cache[1] if cache else None
    inv_scale=k.shape[-1]**-0.5
    q=(inv_scale**2)*mx.fast.rms_norm(q,None,1e-6)
    k=inv_scale*mx.fast.rms_norm(k,None,1e-6)
    # llama.cpp maps value head h to Q/K head ``h % num_k_heads``.
    # mlx-lm's generic GDN helper uses repeat(), which instead groups three
    # copies of each head. Expand cyclically up front so the helper sees equal
    # head counts and does not apply its grouped repeat.
    if self.num_v_heads!=self.num_k_heads:
        repeats=self.num_v_heads//self.num_k_heads
        q=mx.tile(q,(1,1,repeats,1))
        k=mx.tile(k,(1,1,repeats,1))
    out,state=gated_delta_update(
        q,k,v,a,b,self.A_log,self.dt_bias,state,mask,
        use_kernel=not self.training,
    )
    if cache is not None:
        cache[1]=state
        cache.advance(S)
    out=self.norm(out,z)
    out=self.out_proj(out.reshape(B,S,-1))
    if self.sharding_group is not None:
        out=mx.distributed.all_sum(out,group=self.sharding_group)
    return out


@lru_cache(maxsize=1)
def _gguf_gated_delta_class(base):
    return type("GSQGatedDeltaNet",(base,),{"__call__":_gguf_gated_delta_call})


def build_qwen35(store,dtype=mx.float16):
    """Build an mlx-lm TextModel with GSQ layers backed by ``store``."""
    from mlx_lm.models.qwen3_5 import TextModel
    model=TextModel(_args())
    model.model.embed_tokens=GSQEmbedding(store.weight("token_embd.weight"),dtype=dtype)
    model.model.norm.weight=_plain(store,"output_norm.weight").astype(dtype)
    model.lm_head=GSQLinear(store.weight("output.weight"))

    for i,layer in enumerate(model.model.layers):
        prefix=f"blk.{i}."
        layer.input_layernorm.weight=_plain(store,prefix+"attn_norm.weight").astype(dtype)
        layer.post_attention_layernorm.weight=_plain(store,prefix+"post_attention_norm.weight").astype(dtype)
        layer.mlp.gate_proj=GSQLinear(store.weight(prefix+"ffn_gate.weight"))
        layer.mlp.up_proj=GSQLinear(store.weight(prefix+"ffn_up.weight"))
        layer.mlp.down_proj=GSQLinear(store.weight(prefix+"ffn_down.weight"))
        if layer.is_linear:
            a=layer.linear_attn
            a.__class__=_gguf_gated_delta_class(type(a))
            a.in_proj_qkv=GSQLinear(store.weight(prefix+"attn_qkv.weight"))
            a.in_proj_z=GSQLinear(store.weight(prefix+"attn_gate.weight"))
            a.out_proj=GSQLinear(store.weight(prefix+"ssm_out.weight"))
            a.in_proj_a.weight=_plain(store,prefix+"ssm_alpha.weight").astype(dtype)
            a.in_proj_b.weight=_plain(store,prefix+"ssm_beta.weight").astype(dtype)
            conv=_plain(store,prefix+"ssm_conv1d.weight").astype(dtype)
            a.conv1d.weight=conv.reshape((conv.shape[0],conv.shape[1],1))
            # llama.cpp/GGUF stores the transformed recurrence coefficient,
            # whereas mlx-lm applies ``-exp(A_log)`` inside the update kernel.
            a.A_log=_restore_a_log(_plain(store,prefix+"ssm_a"))
            a.dt_bias=_plain(store,prefix+"ssm_dt.bias").astype(dtype)
            a.norm.weight=_plain(store,prefix+"ssm_norm.weight").astype(dtype)
        else:
            a=layer.self_attn
            a.q_proj=GSQLinear(store.weight(prefix+"attn_q.weight"))
            a.k_proj=GSQLinear(store.weight(prefix+"attn_k.weight"))
            a.v_proj=GSQLinear(store.weight(prefix+"attn_v.weight"))
            a.o_proj=GSQLinear(store.weight(prefix+"attn_output.weight"))
            a.q_norm.weight=_plain(store,prefix+"attn_q_norm.weight").astype(dtype)
            a.k_norm.weight=_plain(store,prefix+"attn_k_norm.weight").astype(dtype)
    return model
