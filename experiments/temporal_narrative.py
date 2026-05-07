"""
Experiment 6: Temporal Narrative — Complete Redesign
500-event research log with 3-way overlapping entity disambiguation.
Target requires finding a unique scientist+compound+subject combination among
many partial matches, forcing full-context positional attention.
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

# Entity pools — smaller pools = more overlap = harder disambiguation
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
    """Generate random alphanumeric code."""
    return f"{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{random.randint(1000, 9999)}"


def _make_research_log(num_events: int, target_scientist: str, target_compound: str,
                       target_subject: str, target_code: str, ratio: float) -> str:
    """Build research log where target is hidden among many partial matches."""
    entries = []
    used_triples = set()

    # Build pool of unique triples, ensuring overlaps with target entities
    # We want multiple entries sharing each target entity
    while len(entries) < num_events - 1:
        s = random.choice(SCIENTISTS)
        c = random.choice(COMPOUNDS)
        sub = random.choice(SUBJECTS)
        triple = (s, c, sub)
        if triple in used_triples:
            continue
        used_triples.add(triple)

        # Bias toward creating partial overlaps with target
        # This ensures target entities appear frequently
        if s == target_scientist or c == target_compound or sub == target_subject:
            if random.random() < 0.7:  # 70% chance to keep overlap entries
                code = _generate_code()
                entries.append((s, c, sub, code))
        else:
            code = _generate_code()
            entries.append((s, c, sub, code))

    # Add target entry
    entries.insert(int(ratio * len(entries)), (target_scientist, target_compound, target_subject, target_code))

    # Shuffle everything
    random.shuffle(entries)

    # Format as chronological log
    lines = []
    for i, (s, c, sub, code) in enumerate(entries):
        lines.append(f"Day {i+1}: {s} tested {c} on {sub}. Result: {code}.")

    return "\n".join(lines)


def run_temporal_narrative(
    model_name: str,
    num_events: int,
    num_examples: int,
    out_dir: str,
    depths: List[float] = None,
) -> Dict[str, Any]:
    """Run 500-event research log with 3-way entity disambiguation."""
    ensure_dir(out_dir)

    if depths is None:
        depths = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

    results = {}
    start = time.time()

    for depth in depths:
        logger.info(f"[NARRATIVE] Depth {depth:.1%}")
        preds = []
        for i in tqdm(range(num_examples), desc=f"Narrative {depth:.1%}", leave=False):
            target_scientist = random.choice(SCIENTISTS)
            target_compound = random.choice(COMPOUNDS)
            target_subject = random.choice(SUBJECTS)
            target_code = _generate_code()

            log = _make_research_log(
                num_events, target_scientist, target_compound,
                target_subject, target_code, depth
            )

            prompt = (
                f"Below is a chronological research log with {num_events} experiment entries.\n\n"
                f"{log}\n\n"
                f"Question: What was the result code when {target_scientist} tested "
                f"{target_compound} on {target_subject}? "
                f"Answer with only the code (format: XX-NNNN)."
            )

            ans = generate_text(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                max_new_tokens=15,
            )
            correct = exact_match_score(ans, target_code)
            preds.append({
                "model_answer": ans,
                "correct": correct,
                "expected_code": target_code,
                "depth": depth,
            })

        save_jsonl(os.path.join(out_dir, f"narrative_depth_{depth}.jsonl"), preds)
        acc = compute_accuracy(preds)
        results[depth] = {"accuracy": acc, "predictions": preds}
        logger.info(f"[NARRATIVE] Depth {depth:.1%}: acc={acc:.3f}")

    summary = {
        "experiment": "temporal_narrative",
        "num_events": num_events,
        "num_examples": num_examples,
        "depths": {str(d): results[d]["accuracy"] for d in depths},
        "pbi": position_bias_index(depths, [results[d]["accuracy"] for d in depths]),
        "time_minutes": (time.time() - start) / 60,
    }

    save_json(os.path.join(out_dir, "narrative_summary.json"), summary)
    plot_curve(
        depths,
        [results[d]["accuracy"] for d in depths],
        f"Exp 6: Temporal Narrative ({num_events} events)",
        os.path.join(out_dir, "narrative_curve.png"),
        xlabel="Depth in Document (0=start, 1=end)",
    )

    logger.info(f"[NARRATIVE] Time={(time.time()-start)/60:.1f} min")
    return summary
