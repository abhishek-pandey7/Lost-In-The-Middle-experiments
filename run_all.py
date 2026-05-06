#!/usr/bin/env python3
"""
================================================================================
LOST IN THE MIDDLE — Benchmark Suite v4 (Master Runner)
================================================================================
Runs all 7 experiments with configurable model, counts, and output directory.
Usage:
    python run_all.py --model Qwen/Qwen2.5-1.5B-Instruct --output ./results
================================================================================
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time

from experiments.kv_retrieval import run_kv_retrieval
from experiments.needle_in_haystack import run_needle_in_haystack
from experiments.multi_needle import run_multi_needle
from experiments.fact_reasoning import run_fact_reasoning
from experiments.semantic_distractors import run_semantic_distractors
from experiments.temporal_narrative import run_temporal_narrative
from experiments.conversation_memory import run_conversation_memory
from src.utils import save_json

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="LITM Benchmark Suite v4")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="HF model name")
    p.add_argument("--output", default="./results", help="Output directory")
    p.add_argument("--n-examples", type=int, default=50, help="Examples per position")
    p.add_argument("--n-keys-100", type=int, default=100)
    p.add_argument("--n-keys-200", type=int, default=200)
    p.add_argument("--needle-sentences", type=int, default=500)
    p.add_argument("--multi-sentences", type=int, default=300)
    p.add_argument("--reason-sentences", type=int, default=300)
    p.add_argument("--semantic-facts", type=int, default=80)
    p.add_argument("--narrative-events", type=int, default=100)
    p.add_argument("--convo-turns", type=int, default=100)
    p.add_argument("--experiments", default="all", help="Comma-separated list or 'all'")
    p.add_argument("--zip", action="store_true", help="Create zip archive of results")
    return p.parse_args()


def main():
    args = parse_args()
    model = args.model
    out_root = args.output
    os.makedirs(out_root, exist_ok=True)

    wanted = set(args.experiments.split(",")) if args.experiments != "all" else {"all"}

    logger.info("=" * 70)
    logger.info("LITM BENCHMARK SUITE v4")
    logger.info(f"Model: {model} | Output: {out_root}")
    logger.info("=" * 70)

    all_results = {}
    t0 = time.time()

    def should_run(name):
        return "all" in wanted or name in wanted

    if should_run("kv100"):
        logger.info("\n--- EXP 1A: KV Retrieval (100 keys) ---")
        all_results["kv_100"] = run_kv_retrieval(
            model_name=model,
            num_keys=args.n_keys_100,
            num_examples=args.n_examples,
            out_dir=os.path.join(out_root, "exp1a_kv100"),
            prefix="kv100",
        )

    if should_run("kv200"):
        logger.info("\n--- EXP 1B: KV Retrieval (200 keys) ---")
        all_results["kv_200"] = run_kv_retrieval(
            model_name=model,
            num_keys=args.n_keys_200,
            num_examples=args.n_examples,
            out_dir=os.path.join(out_root, "exp1b_kv200"),
            prefix="kv200",
        )

    if should_run("needle"):
        logger.info("\n--- EXP 2: Needle in Haystack ---")
        all_results["needle"] = run_needle_in_haystack(
            model_name=model,
            num_sentences=args.needle_sentences,
            num_examples=30,
            out_dir=os.path.join(out_root, "exp2_needle"),
        )

    if should_run("multi"):
        logger.info("\n--- EXP 3: Multi-Needle ---")
        all_results["multi"] = run_multi_needle(
            model_name=model,
            num_sentences=args.multi_sentences,
            num_examples=30,
            out_dir=os.path.join(out_root, "exp3_multi"),
        )

    if should_run("reason"):
        logger.info("\n--- EXP 4: Fact-Dependent Reasoning ---")
        all_results["reason"] = run_fact_reasoning(
            model_name=model,
            num_sentences=args.reason_sentences,
            num_examples=30,
            out_dir=os.path.join(out_root, "exp4_reason"),
        )

    if should_run("semantic"):
        logger.info("\n--- EXP 5: Semantic Similarity Distractors ---")
        all_results["semantic"] = run_semantic_distractors(
            model_name=model,
            num_facts=args.semantic_facts,
            num_examples=30,
            out_dir=os.path.join(out_root, "exp5_semantic"),
        )

    if should_run("narrative"):
        logger.info("\n--- EXP 6: Temporal Narrative ---")
        all_results["narrative"] = run_temporal_narrative(
            model_name=model,
            num_events=args.narrative_events,
            num_examples=30,
            out_dir=os.path.join(out_root, "exp6_narrative"),
        )

    if should_run("conversation"):
        logger.info("\n--- EXP 7: Conversation Memory ---")
        all_results["conversation"] = run_conversation_memory(
            model_name=model,
            num_turns=args.convo_turns,
            num_examples=30,
            out_dir=os.path.join(out_root, "exp7_conversation"),
        )

    elapsed = (time.time() - t0) / 3600
    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE. Total time: {elapsed:.2f} hours")
    logger.info(f"Results: {out_root}")
    logger.info(f"{'='*70}")

    save_json(os.path.join(out_root, "master_summary.json"), all_results)

    # Print PBI table
    logger.info("\n--- Position Bias Index (PBI) Summary ---")
    for k, v in all_results.items():
        if isinstance(v, dict) and "pbi" in v:
            logger.info(f"  {k:20s} PBI = {v['pbi']:+.3f}")

    if args.zip:
        zip_path = os.path.join(os.path.dirname(out_root), "litm_results_all")
        shutil.make_archive(zip_path, "zip", out_root)
        logger.info(f"Zipped: {zip_path}.zip")


if __name__ == "__main__":
    main()
