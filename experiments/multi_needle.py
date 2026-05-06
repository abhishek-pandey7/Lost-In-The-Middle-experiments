"""
Experiment 3: Multi-Needle Retrieval
Tests ability to retrieve ALL of multiple needles placed at start, middle, and end.
"""
import logging
import os
import random
import time
from typing import Dict, Any

from tqdm import tqdm

from src.generator import generate_text
from src.metrics import exact_match_score, compute_accuracy
from src.plotting import plot_bar
from src.utils import ensure_dir, save_json

logger = logging.getLogger(__name__)

from .needle_in_haystack import FILLERS


def _make_haystack(n: int) -> str:
    return " ".join(random.choice(FILLERS) + f" [{i+1}]." for i in range(n))


def run_multi_needle(
    model_name: str,
    num_sentences: int,
    num_examples: int,
    out_dir: str,
) -> Dict[str, Any]:
    """Run multi-needle experiment."""
    ensure_dir(out_dir)

    start = time.time()
    start_ok, mid_ok, end_ok = [], [], []

    for i in tqdm(range(num_examples), desc="Multi-needle"):
        filler = _make_haystack(num_sentences)
        sents = [s.strip() + "." for s in filler.split(".") if s.strip()]
        n = len(sents)
        ca, cb, cc = f"ALPHA-{i:03d}", f"BETA-{i:03d}", f"GAMMA-{i:03d}"

        sents.insert(0, f"The first secret code is {ca}.")
        sents.insert(n // 2, f"The second secret code is {cb}.")
        sents.append(f"The third secret code is {cc}.")

        prompt = (
            f"Read the text and list ALL three secret codes in order.\n\n"
            f"{' '.join(sents)}\n\nCodes:"
        )
        ans = generate_text(
            [{"role": "user", "content": prompt}],
            model_name=model_name,
            max_new_tokens=60,
        )
        start_ok.append(exact_match_score(ans, ca))
        mid_ok.append(exact_match_score(ans, cb))
        end_ok.append(exact_match_score(ans, cc))

    summary = {
        "experiment": "multi_needle",
        "num_sentences": num_sentences,
        "num_examples": num_examples,
        "start": compute_accuracy([{"correct": c} for c in start_ok]),
        "middle": compute_accuracy([{"correct": c} for c in mid_ok]),
        "end": compute_accuracy([{"correct": c} for c in end_ok]),
        "time_minutes": (time.time() - start) / 60,
    }

    logger.info(
        f"[MULTI] Start={summary['start']:.3f} Mid={summary['middle']:.3f} End={summary['end']:.3f}"
    )

    save_json(os.path.join(out_dir, "multi_summary.json"), summary)
    plot_bar(
        ["Start", "Middle", "End"],
        [summary["start"], summary["middle"], summary["end"]],
        f"Exp 3: Multi-Needle (n={num_examples})",
        os.path.join(out_dir, "multi_bar.png"),
    )

    return summary
