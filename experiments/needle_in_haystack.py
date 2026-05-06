"""
Experiment 2: Needle in Haystack (text)
Tests retrieval of a secret code hidden at varying depths in filler text.
"""
import logging
import os
import random
import time
from typing import List, Dict, Any

from tqdm import tqdm

from src.generator import generate_text
from src.metrics import exact_match_score, compute_accuracy, position_bias_index
from src.plotting import plot_curve
from src.utils import ensure_dir, save_jsonl, save_json

logger = logging.getLogger(__name__)

FILLERS = [
    "The history of pottery spans thousands of years.",
    "Marine biologists study coral reef ecosystems.",
    "Railway engineering requires precise curvature.",
    "The periodic table arranges elements by number.",
    "Clouds are classified as cumulus and stratus.",
    "Beekeeping traditions differ between continents.",
    "The Great Wall was built over many dynasties.",
    "Thermodynamics governs heat transfer.",
    "Impressionist painters captured fleeting light.",
    "Volcanic activity is tracked with seismographs.",
    "The Dewey Decimal System organizes libraries.",
    "Irrigation evolved from canals to drip systems.",
    "Neural networks are inspired by biological brains.",
    "Light speed is 299,792,458 meters per second.",
    "Classical composition follows harmonic rules.",
    "Urban planning addresses zoning and transport.",
    "Photosynthesis converts CO2 into glucose.",
    "The Fibonacci sequence appears in nature.",
    "GPS uses triangulation from satellites.",
    "Cryptography secures digital communication.",
]


def _make_haystack(n: int) -> str:
    """Generate n sentences of filler text."""
    return " ".join(random.choice(FILLERS) + f" [{i+1}]." for i in range(n))


def _insert_needle(text: str, needle: str, ratio: float) -> str:
    """Insert needle at specified depth ratio."""
    sents = [s.strip() + "." for s in text.split(".") if s.strip()]
    idx = int(ratio * len(sents))
    sents.insert(idx, needle)
    return " ".join(sents)


def run_needle_in_haystack(
    model_name: str,
    num_sentences: int,
    num_examples: int,
    out_dir: str,
    depths: List[float] = None,
) -> Dict[str, Any]:
    """Run needle-in-haystack experiment."""
    ensure_dir(out_dir)

    if depths is None:
        depths = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

    results = {}
    start = time.time()

    for depth in depths:
        logger.info(f"[NEEDLE] Depth {depth:.1%}")
        preds = []
        for i in tqdm(range(num_examples), desc=f"Needle {depth:.1%}", leave=False):
            filler = _make_haystack(num_sentences)
            code = f"SECRET-{i:04d}"
            needle = f"The secret code is {code}."
            text = _insert_needle(filler, needle, depth)
            prompt = (
                f"Read the text and find the secret code.\n\n{text}\n\n"
                f"What is the secret code? Answer with only the code."
            )
            ans = generate_text(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                max_new_tokens=20,
            )
            correct = exact_match_score(ans, code)
            preds.append({
                "model_answer": ans,
                "correct": correct,
                "secret": code,
                "depth": depth,
            })

        save_jsonl(os.path.join(out_dir, f"needle_depth_{depth}.jsonl"), preds)
        acc = compute_accuracy(preds)
        results[depth] = {"accuracy": acc, "predictions": preds}
        logger.info(f"[NEEDLE] Depth {depth:.1%}: acc={acc:.3f}")

    summary = {
        "experiment": "needle_in_haystack",
        "num_sentences": num_sentences,
        "num_examples": num_examples,
        "depths": {str(d): results[d]["accuracy"] for d in depths},
        "pbi": position_bias_index(depths, [results[d]["accuracy"] for d in depths]),
        "time_minutes": (time.time() - start) / 60,
    }

    save_json(os.path.join(out_dir, "needle_summary.json"), summary)
    plot_curve(
        depths,
        [results[d]["accuracy"] for d in depths],
        f"Exp 2: Needle in Haystack ({num_sentences} sentences)",
        os.path.join(out_dir, "needle_curve.png"),
        xlabel="Depth in Document (0=start, 1=end)",
    )

    logger.info(f"[NEEDLE] Time={(time.time()-start)/60:.1f} min")
    return summary
