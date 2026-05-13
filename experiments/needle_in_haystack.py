"""
Experiment 2: Needle in Haystack (text)
Tests retrieval of a secret code hidden at varying depths in filler text.
OPTIMAL CONFIG for 1.5B model: 500 sentences + 5 same-format decoys.
Produces asymmetric LITM: strong primacy, weak recency (PBI ≈ +0.2).
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

# Fixed pool of 5 decoy codes with same CL- prefix, different values
DECOY_CODES = [f"CL-{random.randint(1000, 9999)}" for _ in range(5)]


def _make_haystack(n: int, target_code: str, depth: float, num_decoys: int = 5) -> str:
    """Generate haystack with same-prefix decoys scattered throughout."""
    sents = []
    
    # Place decoys at random positions, keeping them away from target
    target_idx = int(depth * n)
    available = [i for i in range(n) if abs(i - target_idx) > 20]
    
    if num_decoys > 0 and len(available) >= num_decoys:
        decoy_positions = set(random.sample(available, num_decoys))
    else:
        decoy_positions = set()
    
    for i in range(n):
        if i in decoy_positions:
            sents.append(f"The classified identifier is {random.choice(DECOY_CODES)}.")
        else:
            sents.append(random.choice(FILLERS))
    
    # Insert target at exact depth
    idx = int(depth * len(sents))
    sents.insert(idx, f"The classified identifier is {target_code}.")
    return " ".join(sents)


def run_needle_in_haystack(
    model_name: str,
    num_sentences: int,
    num_examples: int,
    out_dir: str,
    num_decoys: int = 5,
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
            code = f"CL-{random.randint(10000, 99999)}"
            text = _make_haystack(num_sentences, code, depth, num_decoys)
            
            prompt = (
                f"Read the text carefully. Several codes appear, but only one is "
                f"the true classified identifier. Find it.\n\n{text}\n\n"
                f"What is the classified identifier? Answer with only the code (including CL- prefix)."
            )
            ans = generate_text(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                max_new_tokens=15,
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
        "num_decoys": num_decoys,
        "num_examples": num_examples,
        "depths": {str(d): results[d]["accuracy"] for d in depths},
        "pbi": position_bias_index(depths, [results[d]["accuracy"] for d in depths]),
        "time_minutes": (time.time() - start) / 60,
    }

    save_json(os.path.join(out_dir, "needle_summary.json"), summary)
    plot_curve(
        depths,
        [results[d]["accuracy"] for d in depths],
        f"Exp 2: Needle in Haystack ({num_sentences} sentences, {num_decoys} decoys)",
        os.path.join(out_dir, "needle_curve.png"),
        xlabel="Depth in Document (0=start, 1=end)",
    )

    logger.info(f"[NEEDLE] Time={(time.time()-start)/60:.1f} min")
    return summary
