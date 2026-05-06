"""
Experiment 5: Semantic Similarity Distractors
Gold fact ("capital of France is Paris") among semantically similar facts.
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

TEMPLATES = [
    "The capital of {country} is {city}.",
    "The population of {country} is approximately {num} million.",
    "The official language of {country} is {lang}.",
    "The currency of {country} is the {currency}.",
    "The largest city in {country} is {city}.",
]

COUNTRIES = [
    "Germany", "Spain", "Italy", "Brazil", "Argentina", "Canada",
    "Australia", "Japan", "China", "India", "Russia", "Egypt",
    "Turkey", "Mexico", "South Korea", "Thailand", "Vietnam",
    "Poland", "Sweden", "Norway", "Denmark", "Finland", "Greece",
    "Portugal", "Ireland", "Austria", "Switzerland", "Belgium",
    "Netherlands", "Czech Republic", "Hungary", "Romania",
]

CITIES = [
    "Berlin", "Madrid", "Rome", "Brasilia", "Buenos Aires", "Ottawa",
    "Canberra", "Tokyo", "Beijing", "New Delhi", "Moscow", "Cairo",
    "Ankara", "Mexico City", "Seoul", "Bangkok", "Hanoi",
    "Warsaw", "Stockholm", "Oslo", "Copenhagen", "Helsinki", "Athens",
    "Lisbon", "Dublin", "Vienna", "Bern", "Brussels",
    "Amsterdam", "Prague", "Budapest", "Bucharest",
]

LANGS = [
    "German", "Spanish", "Italian", "Portuguese", "French",
    "English", "Japanese", "Mandarin", "Hindi", "Russian",
    "Arabic", "Turkish", "Korean", "Thai", "Vietnamese",
    "Polish", "Swedish", "Norwegian", "Danish", "Finnish",
    "Greek", "Irish", "Dutch", "Czech", "Hungarian", "Romanian",
]

CURRENCIES = [
    "Euro", "Peso", "Real", "Dollar", "Yen", "Yuan", "Rupee",
    "Ruble", "Pound", "Won", "Baht", "Dong", "Zloty",
    "Krone", "Krona", "Forint", "Leu", "Franc",
]


def _make_doc(num_facts: int, gold_fact: str, ratio: float) -> str:
    facts = []
    for _ in range(num_facts):
        t = random.choice(TEMPLATES)
        fact = t.format(
            country=random.choice(COUNTRIES),
            city=random.choice(CITIES),
            num=random.randint(10, 1400),
            lang=random.choice(LANGS),
            currency=random.choice(CURRENCIES),
        )
        facts.append(fact)

    idx = int(ratio * len(facts))
    facts.insert(idx, gold_fact)
    return "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))


def run_semantic_distractors(
    model_name: str,
    num_facts: int,
    num_examples: int,
    out_dir: str,
    depths: List[float] = None,
) -> Dict[str, Any]:
    """Run semantic distractor experiment."""
    ensure_dir(out_dir)

    if depths is None:
        depths = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

    results = {}
    start = time.time()

    for depth in depths:
        logger.info(f"[SEMANTIC] Depth {depth:.1%}")
        preds = []
        for i in tqdm(range(num_examples), desc=f"Semantic {depth:.1%}", leave=False):
            gold = "The capital of France is Paris."
            doc = _make_doc(num_facts, gold, depth)
            prompt = (
                f"Read the following list of facts and answer the question.\n\n{doc}\n\n"
                f"Question: What is the capital of France? Answer with only the city name."
            )
            ans = generate_text(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                max_new_tokens=20,
            )
            correct = exact_match_score(ans, "paris")
            preds.append({
                "model_answer": ans,
                "correct": correct,
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
