"""
Experiment 6: Temporal Narrative
Recall an event from a long chronological timeline.
"""
import logging
import os
import random
import re
import time
from typing import List, Dict, Any

from tqdm import tqdm

from src.generator import generate_text
from src.metrics import exact_match_score, compute_accuracy, position_bias_index
from src.plotting import plot_curve
from src.utils import ensure_dir, save_jsonl, save_json

logger = logging.getLogger(__name__)

EVENTS_POOL = [
    "the king issued a decree",
    "a comet appeared in the sky",
    "the bridge was completed",
    "a treaty was signed",
    "the harvest festival began",
    "a stranger arrived at the gates",
    "the library burned down",
    "a new star was discovered",
    "the river flooded the town",
    "the army marched north",
    "a peace envoy was sent",
    "the market was opened",
    "a plague swept the city",
    "the old temple was restored",
    "a fleet set sail for distant lands",
    "the academy admitted its first students",
    "a rebellion broke out in the east",
    "the queen gave birth to twins",
    "a dragon was spotted in the mountains",
    "the great bell tolled for the first time",
]


def _make_timeline(num_events: int, target_event: str, ratio: float) -> str:
    events = random.sample(EVENTS_POOL, min(num_events, len(EVENTS_POOL)))
    while len(events) < num_events:
        events.append(
            f"the people gathered for the {random.choice(['morning', 'evening', 'midday'])} ceremony"
        )
    idx = int(ratio * len(events))
    events.insert(idx, target_event)
    return "\n".join(f"Year {1000+i}: {e}." for i, e in enumerate(events))


def run_temporal_narrative(
    model_name: str,
    num_events: int,
    num_examples: int,
    out_dir: str,
    depths: List[float] = None,
) -> Dict[str, Any]:
    """Run temporal narrative experiment."""
    ensure_dir(out_dir)

    if depths is None:
        depths = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

    results = {}
    start = time.time()

    for depth in depths:
        logger.info(f"[NARRATIVE] Depth {depth:.1%}")
        preds = []
        for i in tqdm(range(num_examples), desc=f"Narrative {depth:.1%}", leave=False):
            target = "a golden statue was unveiled in the central square"
            timeline = _make_timeline(num_events, target, depth)
            prompt = (
                f"Read the following timeline of historical events.\n\n{timeline}\n\n"
                f"Question: In which year was a golden statue unveiled in the central square? "
                f"Answer with only the year number."
            )
            ans = generate_text(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                max_new_tokens=15,
            )
            expected_year = 1000 + int(depth * num_events)
            years = re.findall(r"\b\d{4}\b", ans)
            correct = 1.0 if any(abs(int(y) - expected_year) < 5 for y in years) else 0.0
            preds.append({
                "model_answer": ans,
                "correct": correct,
                "expected_year": expected_year,
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
        xlabel="Depth in Timeline (0=start, 1=end)",
    )

    logger.info(f"[NARRATIVE] Time={(time.time()-start)/60:.1f} min")
    return summary
