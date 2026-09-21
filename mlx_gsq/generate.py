"""Command-line text generation through mlx-lm."""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(prog="python -m mlx_gsq.generate")
    parser.add_argument("model",type=Path)
    parser.add_argument("--tokenizer",type=Path)
    parser.add_argument("--prompt",required=True)
    parser.add_argument("--max-tokens",type=int,default=64)
    parser.add_argument("--verbose",action="store_true")
    args=parser.parse_args()
    from mlx_gsq import load
    from mlx_lm.generate import generate
    model,tokenizer=load(args.model,args.tokenizer)
    result=generate(model,tokenizer,prompt=args.prompt,max_tokens=args.max_tokens,verbose=args.verbose)
    if not args.verbose: print(result)


if __name__=="__main__": main()
