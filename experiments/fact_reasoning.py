"""
Experiment 4: Fact-Dependent Reasoning
Math problem requiring retrieval of a buried fact.
CRITICAL FIX: Uses fictional products and random prices so the model
CANNOT answer from parametric knowledge — it MUST read the document.
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

# Fictional product names — never seen in pretraining
FICTIONAL_PRODUCTS = [
    "Zylor apples", "Krynn berries", "Xylor pears", "Freloria grapes",
    "Vortis melons", "Zenthar plums", "Kandor peaches", "Eldoria cherries",
    "Thaloria figs", "Nyxon limes", "Pyraxis kiwis", "Oblivion mangoes",
    "Cresthaven papayas", "Velmora guavas", "Drakonia dates",
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
    """Run fact-dependent reasoning experiment with fictional products."""
    ensure_dir(out_dir)

    if depths is None:
        depths = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

    results = {}
    start = time.time()

    for depth in depths:
        logger.info(f"[REASON] Depth {depth:.1%}")
        preds = []
        for _ in tqdm(range(num_examples), desc=f"Reason {depth:.1%}", leave=False):
            # Random fictional product with random price (50-500, clearly fictional)
            product = random.choice(FICTIONAL_PRODUCTS)
            price = random.randint(50, 500)
            qty = random.randint(3, 20)
            answer = price * qty  # Simple integer multiplication

            fact = f"For this order, {product} cost ${price}/kg."
            doc = _make_doc(num_sentences, fact, depth)
            prompt = (
                f"Use ONLY the document below. Do not use any outside knowledge.\n\n"
                f"{doc}\n\n"
                f"According to the document, I buy {qty} kg of {product}. "
                f"What is my total cost? Answer with only the number (no dollar sign, no units)."
            )
            ans = generate_text(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                max_new_tokens=20,
            )
            # Extract first integer from answer
            nums = re.findall(r"\b\d+\b", ans.replace(",", ""))
            pred = int(nums[0]) if nums else -1
            correct = 1.0 if pred == answer else 0.0
            preds.append({
                "model_answer": ans,
                "predicted": pred,
                "correct_answer": answer,
                "correct": correct,
                "price": price,
                "quantity": qty,
                "product": product,
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
