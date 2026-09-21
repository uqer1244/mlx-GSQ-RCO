"""Dependency-free GGUF v2/v3 directory reader.

Parses metadata and tensor locations without dequantizing or loading model
payloads into memory.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path
import struct
from typing import Any, BinaryIO


class GGUFError(ValueError):
    pass


class GGUFValueType(IntEnum):
    UINT8=0; INT8=1; UINT16=2; INT16=3; UINT32=4; INT32=5; FLOAT32=6
    BOOL=7; STRING=8; ARRAY=9; UINT64=10; INT64=11; FLOAT64=12


class GGMLQuantizationType(IntEnum):
    F32=0; F16=1; Q4_0=2; Q4_1=3; Q5_0=6; Q5_1=7; Q8_0=8; Q8_1=9
    Q2_K=10; Q3_K=11; Q4_K=12; Q5_K=13; Q6_K=14; Q8_K=15
    IQ2_XXS=16; IQ2_XS=17; IQ3_XXS=18; IQ1_S=19; IQ4_NL=20
    IQ3_S=21; IQ2_S=22; IQ4_XS=23; I8=24; I16=25; I32=26; I64=27
    F64=28; IQ1_M=29; BF16=30; TQ1_0=34; TQ2_0=35; MXFP4=39; NVFP4=40


# logical elements per block, stored bytes per block (ggml type traits)
QUANT_BLOCK_SIZES = {
    0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),
    8:(32,34),9:(32,36),10:(256,84),11:(256,110),12:(256,144),
    13:(256,176),14:(256,210),15:(256,292),16:(256,66),17:(256,74),
    18:(256,98),19:(256,50),20:(32,18),21:(256,110),22:(256,82),
    23:(256,136),24:(1,1),25:(1,2),26:(1,4),27:(1,8),28:(1,8),
    29:(256,56),30:(1,2),34:(256,54),35:(256,66),39:(32,17),40:(64,36),
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    quantization_type: GGMLQuantizationType
    offset: int
    absolute_offset: int
    num_elements: int
    num_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gguf_shape": list(self.shape),
            "logical_shape": list(reversed(self.shape)),
            "quantization_type": self.quantization_type.name,
            "quantization_type_id": int(self.quantization_type),
            "offset": self.offset,
            "absolute_offset": self.absolute_offset,
            "num_elements": self.num_elements,
            "num_bytes": self.num_bytes,
        }


class _Decoder:
    formats = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
    def __init__(self, stream: BinaryIO): self.stream = stream
    def unpack(self, fmt: str) -> Any:
        size = struct.calcsize("<" + fmt); raw = self.stream.read(size)
        if len(raw) != size: raise GGUFError("unexpected end of file")
        return struct.unpack("<" + fmt, raw)[0]
    def string(self) -> str:
        length = self.unpack("Q"); raw = self.stream.read(length)
        if len(raw) != length: raise GGUFError("unexpected end of GGUF string")
        return raw.decode("utf-8")
    def value(self, kind: GGUFValueType) -> Any:
        if kind.value in self.formats: return self.unpack(self.formats[kind.value])
        if kind == GGUFValueType.STRING: return self.string()
        if kind == GGUFValueType.ARRAY:
            subtype = GGUFValueType(self.unpack("I"))
            return [self.value(subtype) for _ in range(self.unpack("Q"))]
        raise GGUFError(f"unsupported metadata type {kind}")


class GGUFReader:
    """Read and validate GGUF metadata and tensor extents."""
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve(); self.file_size = self.path.stat().st_size
        with self.path.open("rb") as stream:
            d = _Decoder(stream)
            if stream.read(4) != b"GGUF": raise GGUFError("not a GGUF file")
            self.version = d.unpack("I")
            if self.version not in (2,3): raise GGUFError(f"unsupported GGUF version {self.version}")
            tensor_count, metadata_count = d.unpack("Q"), d.unpack("Q")
            self.metadata = {}
            for _ in range(metadata_count):
                key = d.string()
                if key in self.metadata: raise GGUFError(f"duplicate metadata key {key}")
                self.metadata[key] = d.value(GGUFValueType(d.unpack("I")))
            raw_tensors=[]; names=set()
            for _ in range(tensor_count):
                name=d.string()
                if name in names: raise GGUFError(f"duplicate tensor {name}")
                names.add(name)
                shape=tuple(d.unpack("Q") for _ in range(d.unpack("I")))
                raw_tensors.append((name,shape,GGMLQuantizationType(d.unpack("I")),d.unpack("Q")))
            alignment=self.metadata.get("general.alignment",32)
            if not isinstance(alignment,int) or alignment < 1: raise GGUFError("invalid alignment")
            self.data_offset=(stream.tell()+alignment-1)//alignment*alignment
        tensors=[]; ranges=[]
        for name,shape,qtype,offset in raw_tensors:
            count=1
            for dimension in shape: count*=dimension
            block,size=QUANT_BLOCK_SIZES[int(qtype)]
            if count%block: raise GGUFError(f"invalid {qtype.name} element count for {name}")
            num_bytes=count//block*size; absolute=self.data_offset+offset
            if absolute+num_bytes>self.file_size: raise GGUFError(f"tensor outside file: {name}")
            ranges.append((absolute,absolute+num_bytes,name))
            tensors.append(TensorInfo(name,shape,qtype,offset,absolute,count,num_bytes))
        ordered=sorted(ranges)
        for left,right in zip(ordered,ordered[1:]):
            if left[1]>right[0]: raise GGUFError(f"overlapping tensors: {left[2]}, {right[2]}")
        self.tensors=tuple(tensors); self._by_name={t.name:t for t in tensors}

    def tensor(self,name: str)->TensorInfo: return self._by_name[name]

    def analysis(self)->dict[str,Any]:
        dist=defaultdict(lambda:{"tensor_count":0,"parameter_count":0,"byte_count":0})
        for t in self.tensors:
            row=dist[t.quantization_type.name]; row["tensor_count"]+=1
            row["parameter_count"]+=t.num_elements; row["byte_count"]+=t.num_bytes
        metadata={k:({"kind":"array","length":len(v)} if isinstance(v,list) and len(v)>256 else v) for k,v in self.metadata.items()}
        return {
            "schema_version":1,
            "source":{"path":str(self.path),"file_size":self.file_size,"gguf_version":self.version,"data_offset":self.data_offset},
            "metadata":metadata,
            "summary":{"total_tensors":len(self.tensors),"total_parameters":sum(t.num_elements for t in self.tensors),"total_tensor_bytes":sum(t.num_bytes for t in self.tensors)},
            "quantization_distribution":dict(sorted(dist.items())),
            "tensors":[t.as_dict() for t in self.tensors],
        }


class GGUFModelAnalyzer:
    @staticmethod
    def parse(model_path: str|Path)->dict[str,Any]: return GGUFReader(model_path).analysis()


def write_analysis(model_path: str|Path,output_path: str|Path)->dict[str,Any]:
    result=GGUFReader(model_path).analysis()
    Path(output_path).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    return result


def get_tensor_quant_distribution(model_path: str|Path)->dict[str,int]:
    rows=GGUFReader(model_path).analysis()["quantization_distribution"]
    return {name:row["tensor_count"] for name,row in rows.items()}


def get_quant_distribution_with_params(model_path: str|Path)->dict[str,dict[str,int]]:
    return GGUFReader(model_path).analysis()["quantization_distribution"]
