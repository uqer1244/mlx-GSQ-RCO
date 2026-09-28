"""QMM prefill microbenchmark.

Benchmarks the tiled QMM kernel on the model's real large projections
(rows=128 ~ realistic prefill chunk) and reports GFLOPS per qtype.
Also supports a decode-only mode to measure the packed-weight decode cost
as a fraction of total kernel time.
"""
from __future__ import annotations
import sys, time
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from mlx_gsq.runtime import PackedModelStore

MODEL="Qwen3.8-27B-GSQ-RCO/model-packed.safetensors"
# (qtype, N, K) representative of the model's largest projections
CASES=[
    ("IQ2_S",   248320, 5120),   # lm_head / embedding
    ("IQ3_XXS", 5120,  17408),   # ffn_down shape class
    ("IQ3_S",   5120,  17408),
    ("IQ4_XS",  17408,  5120),
    ("IQ2_XXS", 17408,  5120),
    ("IQ2_XS",  17408,  5120),
]
ROWS=128
ITERS=8

def bench(fn, warmup=2):
    for _ in range(warmup):
        o=fn(); mx.eval(o)
    mx.eval(mx.zeros(1))
    t0=time.perf_counter()
    for _ in range(ITERS):
        o=fn(); mx.eval(o)
    mx.eval(mx.zeros(1))
    return (time.perf_counter()-t0)/ITERS

def main():
    store=PackedModelStore(MODEL)
    from mlx_gsq.runtime.kernels.qmm import quantized_matmul
    for qtype,N,K in CASES:
        # find a tensor with exactly this shape
        name=next((n for n,i in store.manifest["tensors"].items()
                   if i["quantization_type"]==qtype and i["logical_shape"][:2]==[N,K]),None)
        if name is None:
            name=next((n for n,i in store.manifest["tensors"].items()
                       if i["quantization_type"]==qtype and i["logical_shape"][0]>=min(N,K)),None)
            w=store.weight(name); N,K=w.logical_shape[0],w.logical_shape[1]
        else:
            w=store.weight(name)
        row_bytes=w.num_bytes//w.logical_shape[0]
        # cap N so the benchmark stays tractable
        Nt=min(N,17408)
        packed=w.data[:Nt*row_bytes]
        rng=np.random.default_rng(0)
        x=mx.array(rng.normal(size=(ROWS,K)).astype(np.float16))
        t=bench(lambda: quantized_matmul(x,packed,qtype=qtype,out_features=Nt,in_features=K))
        flops=2*ROWS*Nt*K
        print(f"{qtype:8s} {name.split('.')[-2][:12]:12s} N={Nt:6d} K={K:6d} rows={ROWS}: {t*1e3:8.2f} ms  {flops/t/1e9:8.1f} GFLOP/s")

if __name__=="__main__":
    main()
