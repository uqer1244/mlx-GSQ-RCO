from pathlib import Path
import json
import struct

import pytest

from mlx_gsq.gguf import GGUFError, GGUFReader, convert_gguf_to_safetensors


def _s(value: str) -> bytes:
    raw=value.encode(); return struct.pack("<Q",len(raw))+raw


def make_tiny(path: Path) -> None:
    raw=bytearray(b"GGUF"+struct.pack("<IQQ",3,1,2))
    raw+=_s("general.architecture")+struct.pack("<I",8)+_s("test")
    raw+=_s("general.alignment")+struct.pack("<II",4,32)
    raw+=_s("weight")+struct.pack("<IQQIQ",2,2,2,0,0)
    raw+=bytes((-len(raw))%32)+struct.pack("<ffff",1,2,3,4)
    path.write_bytes(raw)


def test_parse_and_lossless_convert(tmp_path: Path) -> None:
    source=tmp_path/"tiny.gguf"; output=tmp_path/"tiny.safetensors"; make_tiny(source)
    reader=GGUFReader(source); tensor=reader.tensor("weight")
    assert tensor.shape==(2,2) and tensor.num_bytes==16
    convert_gguf_to_safetensors(source,output)
    with output.open("rb") as stream:
        header_size=struct.unpack("<Q",stream.read(8))[0]
        header=json.loads(stream.read(header_size)); payload=stream.read()
    assert header["weight"]["dtype"]=="U8"
    assert payload==struct.pack("<ffff",1,2,3,4)


def test_bad_magic(tmp_path: Path) -> None:
    path=tmp_path/"bad"; path.write_bytes(b"bad")
    with pytest.raises(GGUFError): GGUFReader(path)


def test_real_model() -> None:
    path=Path("Qwen3.8-27B-GSQ-RCO/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf")
    if not path.exists(): pytest.skip("local model absent")
    reader=GGUFReader(path); analysis=reader.analysis()
    assert reader.metadata["general.architecture"]=="qwen35"
    assert analysis["summary"]["total_tensors"]==866
    assert analysis["summary"]["total_parameters"]==27_320_697_856
    assert analysis["quantization_distribution"]["IQ3_S"]["tensor_count"]==97
