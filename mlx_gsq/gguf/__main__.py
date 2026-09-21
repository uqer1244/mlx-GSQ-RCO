from __future__ import annotations
import argparse
from pathlib import Path
from .converter import convert_gguf_to_safetensors, verify_conversion
from .parser import write_analysis


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mlx_gsq.gguf")
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze"); analyze.add_argument("model", type=Path); analyze.add_argument("-o", "--output", type=Path, default=Path("analysis.json"))
    convert = sub.add_parser("convert"); convert.add_argument("model", type=Path); convert.add_argument("output", type=Path); convert.add_argument("--overwrite", action="store_true"); convert.add_argument("--no-verify", action="store_true")
    verify = sub.add_parser("verify"); verify.add_argument("model", type=Path); verify.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "analyze":
        result = write_analysis(args.model, args.output)
        print(f"validated {result['summary']['total_tensors']} tensors -> {args.output}")
    elif args.command == "convert":
        result = convert_gguf_to_safetensors(args.model, args.output, overwrite=args.overwrite, verify=not args.no_verify)
        print(f"wrote {result}")
    else:
        verify_conversion(args.model, args.output); print("verification passed")


if __name__ == "__main__": main()
