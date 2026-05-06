#!/usr/bin/env python3
"""
================================================================================
LITM v4 — Experiment 3: Multi-Needle Retrieval
Standalone Kaggle-ready script.
================================================================================
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.multi_needle import run_multi_needle

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="LITM v4 - Exp 3: Multi-Needle")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n-examples", type=int, default=30)
    p.add_argument("--n-sentences", type=int, default=300)
    p.add_argument("--output", default="/kaggle/working/litm_results")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(args.output, "exp3_multi")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LITM v4 - Experiment 3: Multi-Needle Retrieval")
    logger.info(f"Model: {args.model}")
    logger.info(f"Examples: {args.n_examples} | Sentences: {args.n_sentences}")
    logger.info(f"Output: {out_dir}")
    logger.info("=" * 60)

    result = run_multi_needle(
        model_name=args.model,
        num_sentences=args.n_sentences,
        num_examples=args.n_examples,
        out_dir=out_dir,
    )

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 3 COMPLETE")
    logger.info(f"Start accuracy: {result.get('start', 'N/A')}")
    logger.info(f"Middle accuracy: {result.get('middle', 'N/A')}")
    logger.info(f"End accuracy: {result.get('end', 'N/A')}")
    logger.info(f"Results saved to: {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
