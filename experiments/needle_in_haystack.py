"""
Experiment 2: Needle in Haystack (text)
Tests retrieval of a secret code hidden at varying depths in filler text.
FIXED: Increased default context to 1500 sentences to stress 1.5B model attention.
Also adds decoy codes to prevent trivial keyword-only retrieval.
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
    "The Fibonacci sequence appears in nature.
    "GPS uses triangulation from satellites.",
    "Cryptography secures digital communication.",
    "Aerodynamics explains lift on aircraft wings.",
    "Meteorologists track pressure systems globally.",
    "The Rosetta Stone enabled hieroglyph translation.",
    "Bioluminescence occurs in deep ocean species.",
    "Microprocessors revolutionized personal computing.",
    "Tectonic plates shift gradually over millennia.",
    "The printing press spread knowledge across Europe.",
    "Quantum entanglement defies classical intuition.",
    "Archaeologists excavate buried ancient settlements.",
    "Photosensors detect minute light variations.",
]

# Decoy sentences with codes that use similar patterns
DECOY_TEMPLATES = [
    "The transaction was logged with code TX-{code}.",
    "The batch identifier is BC-{code}.",
    "Session recorded under SC-{code}.",
    "Access granted via AC-{code}.",
    "Error logged as EC-{code}.",
    "Debug trace shows DC-{code}.",
    "Network packet tagged NC-{code}.",
    "User authenticated with UC-{code}.",
    "System heartbeat code SY-{code}.",
    "Database query ID DB-{code}.",
]


def _make_haystack(n: int, num_decoys: int = 15) -> str:
    """Generate n sentences of filler text with decoy codes scattered throughout."""
    sents = []
    for i in range(n):
        if random.random() < (num_decoys / n) and num_decoys > 0:
            # Insert a decoy sentence instead of filler
            template = random.choice(DECOY_TEMPLATES)
            code = f"{random.randint(1000, 9999)}"
            sents.append(template.format(code=code))
            num_decoys -= 1
        else:
            sents.append(random.choice(FILLERS) + f" [{i+1}].")
    random.shuffle(sents)
    return " ".join(sents)


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
    """Run needle-in-haystack experiment with decoy codes."""
    ensure_dir(out_dir)

    if depths is None:
        depths = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

    results = {}
    start = time.time()

    for depth in depths:
        logger.info(f"[NEEDLE] Depth {depth:.1%}")
        preds = []
        for i in tqdm(range(num_examples), desc=f"Needle {depth:.1%}", leave=False):
            filler = _make_haystack(num_sentences, num_decoys=15)
            code = f"{random.randint(10000, 99999)}"
            # Target uses a different prefix than decoys to make it distinguishable
            # but the model must find it among many codes
            needle = f"The classified identifier is CL-{code}."
            text = _insert_needle(filler, needle, depth)
            prompt = (
                f"Read the text carefully. Multiple codes appear, but only one is "
                f"the classified identifier. Find it.\n\n{text}\n\n"
                f"What is the classified identifier? Answer with only the full code (including CL- prefix)."
            )
            ans = generate_text(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                max_new_tokens=20,
            )
            correct = exact_match_score(ans, f"CL-{code}")
            preds.append({
                "model_answer": ans,
                "correct": correct,
                "secret": f"CL-{code}",
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
