"""
Experiment 1: Key-Value Retrieval
Replicates Liu et al. (2023) with expanded position granularity.
Generates UUID key-value pairs, places gold pair at controlled depths.
"""
import json
import logging
import os
import random
import time
import uuid
from typing import List, Dict, Any

from tqdm import tqdm

from src.generator import generate_text
from src.metrics import exact_match_score, compute_accuracy, position_bias_index
from src.plotting import plot_curve
from src.utils import ensure_dir, save_jsonl, save_json

logger = logging.getLogger(__name__)


def _gen_kv_data(num_keys: int, num_examples: int) -> List[Dict[str, Any]]:
    """Generate key-value pair examples."""
    examples = []
    for _ in tqdm(range(num_examples), desc=f"Gen KV data ({num_keys} keys)"):
        kv = {}
        while len(kv) != num_keys:
            kv[str(uuid.uuid4())] = str(uuid.uuid4())
        ordered = list(kv.items())
        gold = random.choice(ordered)
        examples.append({"ordered_kv_records": ordered, "key": gold[0], "value": gold[1]})
    return examples


def _format_prompt(data: List[tuple], key: str) -> str:
    """Format KV data into prompt template."""
    template = """Extract the value corresponding to the specified key in the JSON object below.

JSON data:
{formatted}

Key: "{key}"
Corresponding value:"""
    formatted = ""
    for i, (k, v) in enumerate(data):
        sc = "{" if i == 0 else " "
        ec = ",\n" if i != len(data) - 1 else "}"
        formatted += sc + f'"{k}": "{v}"' + ec
    return template.format(formatted=formatted, key=key)


def _reorder(example: Dict[str, Any], gold_pos: int) -> Dict[str, Any]:
    """Move gold pair to specified position."""
    ordered = example["ordered_kv_records"]
    key = example["key"]
    value = example["value"]
    gi = next(i for i, (k, v) in enumerate(ordered) if k == key)
    new = ordered[:gi] + ordered[gi + 1:]
    new = new[:gold_pos] + [(key, value)] + new[gold_pos:]
    return {"ordered_kv_records": new, "key": key, "value": value}


def run_kv_retrieval(
    model_name: str,
    num_keys: int,
    num_examples: int,
    out_dir: str,
    positions: List[int] = None,
    prefix: str = "kv",
) -> Dict[str, Any]:
    """
    Run KV retrieval experiment.

    Args:
        model_name: HF model identifier
        num_keys: Number of KV pairs
        num_examples: Examples per position
        out_dir: Output directory
        positions: Custom position list (default: 9 positions)
        prefix: Filename prefix

    Returns:
        Summary dict with accuracy per position and PBI
    """
    ensure_dir(out_dir)

    if positions is None:
        positions = sorted(set([
            0,
            num_keys // 8,
            num_keys // 4,
            3 * num_keys // 8,
            num_keys // 2,
            5 * num_keys // 8,
            3 * num_keys // 4,
            7 * num_keys // 8,
            num_keys - 1,
        ]))

    # Generate data once, then reorder for each position
    data_path = os.path.join(out_dir, f"{prefix}_data.jsonl")
    examples = _gen_kv_data(num_keys, num_examples)
    save_jsonl(data_path, examples)

    results = {}
    start = time.time()

    for pos in positions:
        logger.info(f"[{prefix}] Position {pos}/{num_keys - 1}")
        preds = []
        for ex in tqdm(examples, desc=f"{prefix} pos={pos}", leave=False):
            ro = _reorder(ex, pos)
            prompt = _format_prompt(ro["ordered_kv_records"], ro["key"])
            ans = generate_text(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                max_new_tokens=80,
            )
            correct = exact_match_score(ans, ro["value"])
            preds.append({
                "model_answer": ans,
                "correct": correct,
                "value": ro["value"],
                "gold_position": pos,
            })

        save_jsonl(os.path.join(out_dir, f"{prefix}_pos_{pos}.jsonl"), preds)
        acc = compute_accuracy(preds)
        results[pos] = {"accuracy": acc, "predictions": preds}
        logger.info(f"[{prefix}] Pos {pos}: acc={acc:.3f}")

    # Summary
    norm_pos = [p / (num_keys - 1) for p in sorted(results.keys())]
    accs = [results[p]["accuracy"] for p in sorted(results.keys())]
    pbi = position_bias_index(norm_pos, accs)

    summary = {
        "experiment": "kv_retrieval",
        "num_keys": num_keys,
        "num_examples": num_examples,
        "positions": {str(p): results[p]["accuracy"] for p in sorted(results.keys())},
        "pbi": pbi,
        "time_minutes": (time.time() - start) / 60,
    }

    save_json(os.path.join(out_dir, f"{prefix}_summary.json"), summary)
    plot_curve(
        norm_pos, accs,
        f"Exp 1: KV Retrieval ({num_keys} keys)",
        os.path.join(out_dir, f"{prefix}_curve.png"),
    )

    logger.info(f"[{prefix}] PBI={pbi:.3f} | Time={(time.time()-start)/60:.1f} min")
    return summary
