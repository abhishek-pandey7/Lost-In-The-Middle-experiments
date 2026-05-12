#!/usr/bin/env python3
"""
================================================================================
LITM v4 — Experiment 2: Needle in Haystack
Standalone Kaggle-ready script.
OPTIMIZED: 1000 sentences, 5 same-format decoys for 1.5B model.
================================================================================
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.needle_in_haystack import run_needle_in_haystack

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="LITM v4 - Exp 2: Needle in Haystack")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n-examples", type=int, default=30)
    p.add_argument("--n-sentences", type=int, default=1000,
                   help="Number of sentences in haystack (default 1000 for 1.5B LITM)")
    p.add_argument("--n-decoys", type=int, default=5,
                   help="Number of same-format decoy codes (default 5)")
    p.add_argument("--output", default="/kaggle/working/litm_results")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(args.output, "exp2_needle")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LITM v4 - Experiment 2: Needle in Haystack")
    logger.info(f"Model: {args.model}")
    logger.info(f"Examples: {args.n_examples} | Sentences: {args.n_sentences} | Decoys: {args.n_decoys}")
    logger.info(f"Output: {out_dir}")
    logger.info("=" * 60)

    result = run_needle_in_haystack(
        model_name=args.model,
        num_sentences=args.n_sentences,
        num_examples=args.n_examples,
        num_decoys=args.n_decoys,
        out_dir=out_dir,
    )

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 2 COMPLETE")
    logger.info(f"PBI: {result.get('pbi', 'N/A')}")
    logger.info(f"Results saved to: {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
