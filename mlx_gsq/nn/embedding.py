"""GSQ packed embedding layer."""
import mlx.nn as nn

class GSQEmbedding(nn.Module):
    def __init__(self,weight,dtype=None):
        super().__init__()
        if weight.quantization_type!="IQ2_S": raise NotImplementedError(weight.quantization_type)
        self.packed_weight=weight.data; self.num_embeddings=weight.logical_shape[0]; self.dimensions=weight.logical_shape[1]; self.dtype=dtype
    def __call__(self,ids):
        from mlx_gsq.runtime.kernels.embedding import iq2_s_embedding
        return iq2_s_embedding(ids,self.packed_weight,vocab_size=self.num_embeddings,dimensions=self.dimensions,dtype=self.dtype)
