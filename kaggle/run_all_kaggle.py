#!/usr/bin/env python3
"""
================================================================================
LITM v4 — Kaggle Master Runner (All 7 Experiments)
Runs all experiments sequentially with progress tracking.
================================================================================
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    p = argparse.ArgumentParser(description="LITM v4 - Kaggle Full Run")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output", default="/kaggle/working/litm_results")
    p.add_argument("--n-examples", type=int, default=50)
    p.add_argument("--experiments", default="all",
                   help="all | comma-separated: kv100,kv200,needle,multi,reason,semantic,narrative,conversation")
    return p.parse_args()


def main():
    args = parse_args()
    model = args.model
    out_root = args.output
    os.makedirs(out_root, exist_ok=True)

    wanted = set(args.experiments.split(",")) if args.experiments != "all" else {"all"}

    def should_run(name):
        return "all" in wanted or name in wanted

    all_results = {}
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("LITM BENCHMARK SUITE v4 — KAGGLE FULL RUN")
    logger.info(f"Model: {model}")
    logger.info(f"Output: {out_root}")
    logger.info("=" * 70)

    experiments_to_run = [
        ("kv100", "Experiment 1A: KV Retrieval (100 keys)",
         lambda: run_kv_retrieval(model, 100, args.n_examples,
                                   os.path.join(out_root, "exp1a_kv100"), "kv100")),
        ("kv200", "Experiment 1B: KV Retrieval (200 keys)",
         lambda: run_kv_retrieval(model, 200, args.n_examples,
                                   os.path.join(out_root, "exp1b_kv200"), "kv200")),
        ("needle", "Experiment 2: Needle in Haystack",
         lambda: run_needle_in_haystack(model, 500, 30,
                                        os.path.join(out_root, "exp2_needle"))),
        ("multi", "Experiment 3: Multi-Needle",
         lambda: run_multi_needle(model, 300, 30,
                                   os.path.join(out_root, "exp3_multi"))),
        ("reason", "Experiment 4: Fact-Dependent Reasoning",
         lambda: run_fact_reasoning(model, 300, 30,
                                      os.path.join(out_root, "exp4_reason"))),
        ("semantic", "Experiment 5: Semantic Similarity Distractors",
         lambda: run_semantic_distractors(model, 80, 30,
                                           os.path.join(out_root, "exp5_semantic"))),
        ("narrative", "Experiment 6: Temporal Narrative",
         lambda: run_temporal_narrative(model, 100, 30,
                                         os.path.join(out_root, "exp6_narrative"))),
        ("conversation", "Experiment 7: Conversation Memory",
         lambda: run_conversation_memory(model, 100, 30,
                                          os.path.join(out_root, "exp7_conversation"))),
    ]

    for name, label, runner in experiments_to_run:
        if should_run(name):
            logger.info(f"\n{'='*70}")
            logger.info(f"RUNNING: {label}")
            logger.info(f"{'='*70}")
            exp_start = time.time()
            try:
                result = runner()
                all_results[name] = result
                exp_time = (time.time() - exp_start) / 60
                logger.info(f"✓ {label} completed in {exp_time:.1f} min")
            except Exception as e:
                logger.error(f"✗ {label} failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                all_results[name] = {"error": str(e)}

    elapsed = (time.time() - t0) / 3600
    logger.info(f"\n{'='*70}")
    logger.info(f"ALL EXPERIMENTS COMPLETE. Total time: {elapsed:.2f} hours")
    logger.info(f"{'='*70}")

    save_json(os.path.join(out_root, "master_summary.json"), all_results)

    # PBI Summary
    logger.info("\n--- Position Bias Index (PBI) Summary ---")
    for k, v in all_results.items():
        if isinstance(v, dict) and "pbi" in v:
            logger.info(f"  {k:20s} PBI = {v['pbi']:+.3f}")
        elif isinstance(v, dict) and "error" in v:
            logger.info(f"  {k:20s} ERROR")

    # Zip results
    zip_path = os.path.join(os.path.dirname(out_root), "litm_results_all")
    shutil.make_archive(zip_path, "zip", out_root)
    logger.info(f"\nZipped: {zip_path}.zip")
    logger.info("Download from Kaggle Output tab")


if __name__ == "__main__":
    main()
