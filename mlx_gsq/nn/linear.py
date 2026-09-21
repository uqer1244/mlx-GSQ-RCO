"""Packed mixed-quant linear layer."""
from __future__ import annotations
import mlx.nn as nn


class GSQLinear(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.packed_weight=weight.data
        self.bias=bias
        self.in_features=weight.logical_shape[1]
        self.out_features=weight.logical_shape[0]
        self.qtype=weight.quantization_type

    def __call__(self,x):
        from mlx_gsq.runtime.kernels.qmv import quantized_matvec
        y=quantized_matvec(x,self.packed_weight,qtype=self.qtype,out_features=self.out_features,in_features=self.in_features)
        return y if self.bias is None else y+self.bias
