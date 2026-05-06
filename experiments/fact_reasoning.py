"""
Experiment 4: Fact-Dependent Reasoning
Math problem requiring a fact hidden at varying depths.
"""
import logging
import os
import random
import re
import time
from typing import List, Dict, Any

from tqdm import tqdm

from src.generator import generate_text
from src.metrics import numeric_match, compute_accuracy, position_bias_index
from src.plotting import plot_curve
from src.utils import ensure_dir, save_jsonl, save_json

logger = logging.getLogger(__name__)

DISTRACTORS = [
    "The museum opens at 9 AM.",
    "Temperature is recorded hourly.",
    "The container weighs 2,400 kg.",
    "Ordinances ban construction near rivers.",
    "Q3 revenue increased twelve percent.",
    "The database has four million records.",
    "Solar panels generate 45 kWh daily.",
    "The manuscript was translated in the 1800s.",
    "Airport traffic peaks in summer.",
    "The compound melts at 342 Celsius.",
    "Robotic arms have 0.1mm precision.",
    "Fourteen subspecies were identified.",
    "The hall seats 2,800 guests.",
    "Wastewater uses filtration and aeration.",
    "Satellites show drought vegetation.",
]


def _make_doc(n: int, fact: str, ratio: float) -> str:
    sents = [random.choice(DISTRACTORS) + f" [Doc {i+1}]" for i in range(n)]
    idx = int(ratio * len(sents))
    sents.insert(idx, fact)
    return " ".join(sents)


def run_fact_reasoning(
    model_name: str,
    num_sentences: int,
    num_examples: int,
    out_dir: str,
    depths: List[float] = None,
) -> Dict[str, Any]:
    """Run fact-dependent reasoning experiment."""
    ensure_dir(out_dir)

    if depths is None:
        depths = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

    results = {}
    start = time.time()

    for depth in depths:
        logger.info(f"[REASON] Depth {depth:.1%}")
        preds = []
        for i in tqdm(range(num_examples), desc=f"Reason {depth:.1%}", leave=False):
            price = random.randint(2, 15)
            qty = random.randint(3, 20)
            discount = random.randint(5, 30)
            answer = round(price * qty * (1 - discount / 100), 2)
            fact = f"For this order, apples cost ${price}/kg with a {discount}% discount."
            doc = _make_doc(num_sentences, fact, depth)
            prompt = (
                f"Use ONLY the document below.\n\n{doc}\n\n"
                f"Question: I buy {qty} kg of apples. What is my total cost? "
                f"Answer with only the dollar amount."
            )
            ans = generate_text(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                max_new_tokens=30,
            )
            correct = numeric_match(ans, answer, tolerance=0.5)
            preds.append({
                "model_answer": ans,
                "predicted": float(re.findall(r"[\d,]+\.?\d*", ans.replace(",", ""))[0]) if re.findall(r"[\d,]+\.?\d*", ans.replace(",", "")) else -1.0,
                "correct_answer": answer,
                "correct": correct,
                "depth": depth,
            })

        save_jsonl(os.path.join(out_dir, f"reason_depth_{depth}.jsonl"), preds)
        acc = compute_accuracy(preds)
        results[depth] = {"accuracy": acc, "predictions": preds}
        logger.info(f"[REASON] Depth {depth:.1%}: acc={acc:.3f}")

    summary = {
        "experiment": "fact_reasoning",
        "num_sentences": num_sentences,
        "num_examples": num_examples,
        "depths": {str(d): results[d]["accuracy"] for d in depths},
        "pbi": position_bias_index(depths, [results[d]["accuracy"] for d in depths]),
        "time_minutes": (time.time() - start) / 60,
    }

    save_json(os.path.join(out_dir, "reason_summary.json"), summary)
    plot_curve(
        depths,
        [results[d]["accuracy"] for d in depths],
        f"Exp 4: Fact-Dependent Reasoning ({num_sentences} sentences)",
        os.path.join(out_dir, "reason_curve.png"),
        xlabel="Depth in Document (0=start, 1=end)",
        ylabel="Problem-Solving Accuracy",
    )

    logger.info(f"[REASON] Time={(time.time()-start)/60:.1f} min")
    return summary
