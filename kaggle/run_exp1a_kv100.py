#!/usr/bin/env python3
"""
================================================================================
LITM v4 — Experiment 1A: KV Retrieval (100 keys)
Standalone Kaggle-ready script. Run individually or in sequence.
================================================================================
"""
import argparse
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.kv_retrieval import run_kv_retrieval

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="LITM v4 - Exp 1A: KV Retrieval (100 keys)")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="HF model name")
    p.add_argument("--n-examples", type=int, default=50, help="Examples per position")
    p.add_argument("--n-keys", type=int, default=100, help="Number of KV pairs")
    p.add_argument("--output", default="/kaggle/working/litm_results", help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(args.output, "exp1a_kv100")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LITM v4 - Experiment 1A: KV Retrieval (100 keys)")
    logger.info(f"Model: {args.model}")
    logger.info(f"Examples per position: {args.n_examples}")
    logger.info(f"Keys: {args.n_keys}")
    logger.info(f"Output: {out_dir}")
    logger.info("=" * 60)

    result = run_kv_retrieval(
        model_name=args.model,
        num_keys=args.n_keys,
        num_examples=args.n_examples,
        out_dir=out_dir,
        prefix="kv100",
    )

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT 1A COMPLETE")
    logger.info(f"PBI: {result.get('pbi', 'N/A')}")
    logger.info(f"Results saved to: {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
