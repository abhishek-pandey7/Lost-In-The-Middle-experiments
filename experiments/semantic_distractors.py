"""
Experiment 5: Semantic Similarity Distractors
Gold fact among semantically similar facts.
CRITICAL FIX: Uses random secret codes for fictional entities so the model
CANNOT answer from parametric knowledge — it MUST read the document.
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

# Fictional country names (guaranteed unseen in pretraining)
FICTIONAL_COUNTRIES = [
    "Xyloria", "Freloria", "Zenthar", "Vortis", "Kandor", "Eldoria",
    "Thaloria", "Nyxon", "Pyraxis", "Oblivion", "Cresthaven", "Velmora",
    "Drakonia", "Sylvaris", "Morndell", "Aetheron", "Lumaria", "Obsidius",
    "Quorin", "Tarvos", "Yendell", "Braxil", "Krynn", "Solvaris",
    "Mordant", "Vexilon", "Nymbria", "Oryndor", "Phaedra", "Rivenmoor",
]

# Templates that create semantically similar distractors
TEMPLATES = [
    "The capital of {country} is {code}.",
    "The population of {country} is approximately {num} million.",
    "The official language of {country} is {lang}.",
    "The currency of {country} is the {currency}.",
    "The largest city in {country} is {city}.",
]

# Pools for distractor generation
LANGS = ["Xylorian", "Frelorish", "Zentharan", "Vortian", "Kandoric",
         "Eldorian", "Thalorian", "Nyxonian", "Pyraxian", "Oblivian",
         "Cresthavic", "Velmoran", "Drakonic", "Sylvarian", "Morndelic"]

CURRENCIES = ["Xylor", "Frelor", "Zenthar", "Vort", "Kandor",
              "Eldor", "Thalor", "Nyx", "Pyrax", "Obliv",
              "Crest", "Velm", "Drak", "Sylv", "Morn"]


def _generate_code():
    """Generate a random secret code like PARIS-4281 or ZENTH-7392."""
    return f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}-{random.randint(1000, 9999)}"


def _make_doc(num_facts: int, gold_country: str, gold_code: str, ratio: float) -> str:
    """
    Generate a document with semantically similar distractor facts,
    plus the gold fact at the specified position.
    All countries are fictional and all codes are random — no parametric
    knowledge can answer this.
    """
    # Select distractor countries (different from gold)
    available = [c for c in FICTIONAL_COUNTRIES if c != gold_country]
    distractor_countries = random.sample(available, min(num_facts, len(available)))
    while len(distractor_countries) < num_facts:
        distractor_countries.append(random.choice(available))

    facts = []
    for i, country in enumerate(distractor_countries[:num_facts]):
        t = random.choice(TEMPLATES)
        if "capital" in t or "largest city" in t:
            fact = t.format(country=country, code=_generate_code(), city=_generate_code(),
                            num=random.randint(10, 1400), lang=random.choice(LANGS),
                            currency=random.choice(CURRENCIES))
        elif "population" in t:
            fact = t.format(country=country, code=_generate_code(), city=_generate_code(),
                            num=random.randint(10, 1400), lang=random.choice(LANGS),
                            currency=random.choice(CURRENCIES))
        elif "language" in t:
            fact = t.format(country=country, code=_generate_code(), city=_generate_code(),
                            num=random.randint(10, 1400), lang=random.choice(LANGS),
                            currency=random.choice(CURRENCIES))
        elif "currency" in t:
            fact = t.format(country=country, code=_generate_code(), city=_generate_code(),
                            num=random.randint(10, 1400), lang=random.choice(LANGS),
                            currency=random.choice(CURRENCIES))
        else:
            fact = t.format(country=country, code=_generate_code(), city=_generate_code(),
                            num=random.randint(10, 1400), lang=random.choice(LANGS),
                            currency=random.choice(CURRENCIES))
        facts.append(fact)

    # Insert gold fact at target position
    idx = int(ratio * len(facts))
    gold_fact = f"The capital of {gold_country} is {gold_code}."
    facts.insert(idx, gold_fact)
    return "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))


def run_semantic_distractors(
    model_name: str,
    num_facts: int,
    num_examples: int,
    out_dir: str,
    depths: List[float] = None,
) -> Dict[str, Any]:
    """Run semantic distractor experiment with random secret codes."""
    ensure_dir(out_dir)

    if depths is None:
        depths = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

    results = {}
    start = time.time()

    for depth in depths:
        logger.info(f"[SEMANTIC] Depth {depth:.1%}")
        preds = []
        for i in tqdm(range(num_examples), desc=f"Semantic {depth:.1%}", leave=False):
            # Pick a random fictional country as the target
            gold_country = random.choice(FICTIONAL_COUNTRIES)
            gold_code = _generate_code()

            doc = _make_doc(num_facts, gold_country, gold_code, depth)
            prompt = (
                f"Read the following list of facts and answer the question.\n\n{doc}\n\n"
                f"Question: What is the capital of {gold_country}? "
                f"Answer with only the secret code."
            )
            ans = generate_text(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                max_new_tokens=20,
            )
            # Score against the exact secret code (case-insensitive)
            correct = exact_match_score(ans, gold_code)
            preds.append({
                "model_answer": ans,
                "correct": correct,
                "gold_country": gold_country,
                "gold_code": gold_code,
                "depth": depth,
            })

        save_jsonl(os.path.join(out_dir, f"semantic_depth_{depth}.jsonl"), preds)
        acc = compute_accuracy(preds)
        results[depth] = {"accuracy": acc, "predictions": preds}
        logger.info(f"[SEMANTIC] Depth {depth:.1%}: acc={acc:.3f}")

    summary = {
        "experiment": "semantic_distractors",
        "num_facts": num_facts,
        "num_examples": num_examples,
        "depths": {str(d): results[d]["accuracy"] for d in depths},
        "pbi": position_bias_index(depths, [results[d]["accuracy"] for d in depths]),
        "time_minutes": (time.time() - start) / 60,
    }

    save_json(os.path.join(out_dir, "semantic_summary.json"), summary)
    plot_curve(
        depths,
        [results[d]["accuracy"] for d in depths],
        f"Exp 5: Semantic Similarity Distractors ({num_facts} facts)",
        os.path.join(out_dir, "semantic_curve.png"),
        xlabel="Depth in Document (0=start, 1=end)",
    )

    logger.info(f"[SEMANTIC] Time={(time.time()-start)/60:.1f} min")
    return summary
