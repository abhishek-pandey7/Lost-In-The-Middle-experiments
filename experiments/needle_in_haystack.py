"""
Experiment 2: Needle in Haystack (text)
Tests retrieval of a fact hidden at varying depths in filler text.
FIXED: 2000-sentence haystacks + entity-overlap distractors to prevent keyword-only
retrieval. The target entities (person, item, location) each appear in multiple
sentences, forcing the model to attend to the right combination.
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

# Generic filler sentences
FILLERS = [
    "The history of pottery spans thousands of years.",
    "Marine biologists study coral reef ecosystems.",
    "Railway engineering requires precise curvature calculations.",
    "The periodic table arranges elements by atomic number.",
    "Clouds are classified into cumulus and stratus types.",
    "Beekeeping traditions differ significantly between continents.",
    "The Great Wall was constructed over many successive dynasties.",
    "Thermodynamics governs the principles of heat transfer.",
    "Impressionist painters captured fleeting effects of light.",
    "Volcanic activity is closely tracked with seismographs.",
    "The Dewey Decimal System organizes library collections worldwide.",
    "Irrigation technology evolved from canals to drip systems.",
    "Neural networks are directly inspired by biological brains.",
    "Light speed in vacuum is 299,792,458 meters per second.",
    "Classical composition generally follows established harmonic rules.",
    "Urban planning must address zoning and public transport.",
    "Photosynthesis converts carbon dioxide into glucose and oxygen.",
    "The Fibonacci sequence appears frequently throughout nature.",
    "GPS navigation uses triangulation from orbiting satellites.",
    "Cryptography secures modern digital communications against eavesdropping.",
]

# Entities that appear in MULTIPLE sentences — no single entity is unique
NAMES = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Heidi"]
ITEMS = ["bicycle", "laptop", "watch", "camera", "guitar", "sneakers", "backpack", "headphones"]
PLACES = ["downtown shop", "uptown store", "westside mall", "eastside market", "riverside plaza"]


def _make_entity_distractor(person: str, item: str, place: str, used: set) -> str:
    """Create a distractor sentence sharing 1-2 entities with the target but not all 3."""
    templates = [
        "{person} visited the {place} last Tuesday to browse items.",
        "The {place} sells various products including {item}s and accessories.",
        "{person} enjoys using their {item} during weekend activities.",
        "A customer purchased a {item} from the {place} earlier this month.",
        "{person} recommended the {place} to friends and family members.",
        "The {place} had a promotional sale on {item}s last holiday season.",
        "{person} previously owned a different {item} before upgrading.",
        "Shoppers at the {place} often look for quality {item}s.",
    ]
    # Pick a template and substitute with random entities (may overlap)
    tmpl = random.choice(templates)
    p = random.choice(NAMES)
    it = random.choice(ITEMS)
    pl = random.choice(PLACES)
    sent = tmpl.format(person=p, item=it, place=pl)
    # Ensure it shares at least one entity with the target (person, item, place)
    # but not all three (otherwise it's a duplicate target)
    if p == person and it == item and pl == place:
        # Swap one entity to avoid being identical to target
        swap = random.choice(["person", "item", "place"])
        if swap == "person":
            p = random.choice([n for n in NAMES if n != person])
        elif swap == "item":
            it = random.choice([i for i in ITEMS if i != item])
        else:
            pl = random.choice([pl for pl in PLACES if pl != place])
        sent = tmpl.format(person=p, item=it, place=pl)
    return sent


def _make_haystack(n: int, target_person: str, target_item: str, target_place: str, num_distractors: int = 40) -> str:
    """Generate n sentences with entity-overlap distractors scattered throughout."""
    sents = []
    
    # Add distractor sentences that share entities
    for _ in range(num_distractors):
        sents.append(_make_entity_distractor(target_person, target_item, target_place, set()))
    
    # Fill remaining with generic fillers
    while len(sents) < n:
        sents.append(random.choice(FILLERS))
    
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
    """Run needle-in-haystack with entity-overlap distractors."""
    ensure_dir(out_dir)

    if depths is None:
        depths = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

    results = {}
    start = time.time()

    for depth in depths:
        logger.info(f"[NEEDLE] Depth {depth:.1%}")
        preds = []
        for i in tqdm(range(num_examples), desc=f"Needle {depth:.1%}", leave=False):
            # Choose target entities
            person = random.choice(NAMES)
            item = random.choice(ITEMS)
            place = random.choice(PLACES)
            price = random.randint(100, 999)
            
            # Build haystack with entity-overlap distractors
            filler = _make_haystack(num_sentences, person, item, place, num_distractors=40)
            
            # Target sentence (the needle)
            needle = f"{person} purchased a {item} from the {place} for ${price}."
            text = _insert_needle(filler, needle, depth)
            
            # Question forces the model to find the RIGHT combination, not just any mention
            prompt = (
                f"Read the passage carefully. {person} is mentioned several times, "
                f"and the {place} is mentioned several times, and {item}s are mentioned several times. "
                f"Find the specific sentence that says how much {person} paid for a {item} at the {place}. "
                f"Answer with only the dollar amount (no $ sign, no words)."
            )
            
            ans = generate_text(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                max_new_tokens=10,
            )
            correct = exact_match_score(ans, str(price))
            preds.append({
                "model_answer": ans,
                "correct": correct,
                "expected": price,
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
