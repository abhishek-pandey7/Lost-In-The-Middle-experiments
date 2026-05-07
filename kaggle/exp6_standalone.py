#!/usr/bin/env python3
"""
================================================================================
Exp 6 Standalone — Temporal Narrative with 3-Way Entity Disambiguation
This is a completely self-contained script. No git clone needed.
Run directly in Kaggle: !python exp6_standalone.py
================================================================================
"""
import argparse
import json
import logging
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================
SCIENTISTS = [
    "Dr. Vance", "Dr. Chen", "Dr. Patel", "Dr. Okonkwo", "Dr. Sato",
    "Dr. Müller", "Dr. Silva", "Dr. Kim", "Dr. Ivanov", "Dr. Okafor",
    "Dr. Nakamura", "Dr. Andersson", "Dr. Rossi", "Dr. Singh", "Dr. Larsson",
]

COMPOUNDS = [
    "Zylorium", "Kaptosine", "Vexamide", "Novaline", "Triptorex",
    "Calmantide", "Fluxorol", "Bexatrine", "Yondril", "Pentacil",
    "Luminex", "Dorantin", "Quorafin", "Moxilane", "Zephiron",
    "Nexapril", "Tovacil", "Rexomine", "Solvatrix", "Kryonex",
    "Alphadine", "Betanoril", "Gammaxon", "Deltazol", "Epsilamine",
]

SUBJECTS = [f"Subject-{i:02d}" for i in range(1, 21)]


def _generate_code():
    """Random 2-letter + 4-digit code."""
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"{random.choice(letters)}{random.choice(letters)}-{random.randint(1000, 9999)}"


def _make_log(num_events: int, target_s: str, target_c: str, target_sub: str,
              target_code: str, ratio: float) -> str:
    """Build log where target is at EXACT depth among partial matches."""
    used_triples = set()
    entries = []

    # Build distractors with biased overlap
    while len(entries) < num_events - 1:
        s = random.choice(SCIENTISTS)
        c = random.choice(COMPOUNDS)
        sub = random.choice(SUBJECTS)
        triple = (s, c, sub)
        if triple in used_triples:
            continue
        used_triples.add(triple)
        code = _generate_code()
        entries.append(f"Day PLACEHOLDER: {s} tested {c} on {sub}. Result: {code}.")

    # Shuffle distractors ONLY, then insert target at exact depth
    random.shuffle(entries)
    target_entry = f"Day PLACEHOLDER: {target_s} tested {target_c} on {target_sub}. Result: {target_code}."
    idx = int(ratio * len(entries))
    entries.insert(idx, target_entry)

    # Renumber sequentially
    lines = []
    for i, entry in enumerate(entries):
        lines.append(entry.replace("PLACEHOLDER", str(i + 1)))
    return "\n".join(lines)


# =============================================================================
# MODEL LOADING
# =============================================================================
_model_cache = {}
_tokenizer_cache = {}


def load_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name], _tokenizer_cache[model_name]

    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model.eval()
    _model_cache[model_name] = model
    _tokenizer_cache[model_name] = tokenizer
    logger.info("Model loaded.")
    return model, tokenizer


def generate_text(messages, model_name: str, max_new_tokens: int = 15):
    model, tokenizer = load_model(model_name)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


# =============================================================================
# METRICS
# =============================================================================
def exact_match(pred: str, target: str) -> bool:
    return pred.strip().upper() == target.strip().upper()


def compute_accuracy(preds):
    return sum(1 for p in preds if p["correct"]) / len(preds) if preds else 0.0


def position_bias_index(depths, accs):
    """PBI = mean(start+end)/2 - middle."""
    start = accs[0]
    end = accs[-1]
    middle = accs[len(accs) // 2]
    return (start + end) / 2 - middle


# =============================================================================
# MAIN
# =============================================================================
def run_experiment(model_name: str, num_events: int, num_examples: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    depths = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    results = {}
    start = time.time()

    for depth in depths:
        logger.info(f"[EXP6] Depth {depth:.1%}")
        preds = []
        for i in tqdm(range(num_examples), desc=f"Depth {depth:.1%}", leave=False):
            ts = random.choice(SCIENTISTS)
            tc = random.choice(COMPOUNDS)
            tsub = random.choice(SUBJECTS)
            tcode = _generate_code()

            log = _make_log(num_events, ts, tc, tsub, tcode, depth)
            prompt = (
                f"Below is a research log with {num_events} experiment entries.\n\n"
                f"{log}\n\n"
                f"Question: What was the result code when {ts} tested {tc} on {tsub}? "
                f"Answer with only the code (format: XX-NNNN)."
            )
            ans = generate_text([{"role": "user", "content": prompt}], model_name, max_new_tokens=15)
            correct = exact_match(ans, tcode)
            preds.append({"model_answer": ans, "correct": correct, "expected": tcode, "depth": depth})

        acc = compute_accuracy(preds)
        results[depth] = {"accuracy": acc, "predictions": preds}
        logger.info(f"[EXP6] Depth {depth:.1%}: acc={acc:.3f}")

        # Save per-depth results
        with open(os.path.join(out_dir, f"depth_{depth}.jsonl"), "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")

    accs = [results[d]["accuracy"] for d in depths]
    summary = {
        "experiment": "exp6_temporal_narrative",
        "num_events": num_events,
        "num_examples": num_examples,
        "depths": {str(d): results[d]["accuracy"] for d in depths},
        "pbi": position_bias_index(depths, accs),
        "time_minutes": (time.time() - start) / 60,
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("EXP 6 COMPLETE")
    logger.info(f"PBI: {summary['pbi']:+.3f}")
    for d, a in zip(depths, accs):
        logger.info(f"  Depth {d:5.1%}: {a:.3f}")
    logger.info(f"{'='*60}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--num-events", type=int, default=500)
    p.add_argument("--n-examples", type=int, default=30)
    p.add_argument("--output", default="/kaggle/working/exp6_results")
    args = p.parse_args()

    run_experiment(args.model, args.num_events, args.n_examples, args.output)
