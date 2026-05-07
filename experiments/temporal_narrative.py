"""
Experiment 6: Temporal Narrative — Fixed Version
Target event is semantically embedded among similar distractor events.
A random secret code is attached to the target; the model must distinguish
the target from similar statue-unveiling events and extract the code.
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

# Generic historical events (NOT about statues)
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
    "the northern wall was reinforced",
    "a foreign ambassador visited the court",
    "the mines collapsed unexpectedly",
    "a famous painter completed a masterpiece",
    "the cathedral's dome was finished",
    "a new trade route was established",
    "the royal gardens were opened to the public",
    "a solar eclipse darkened the kingdom",
    "the harbor was expanded",
    "a legendary sword was forged",
]

# Statue-unveiling distractors — same semantic family as target
STATUE_EVENTS = [
    "a bronze statue was unveiled in the town square",
    "a marble statue was unveiled in the village square",
    "a silver statue was unveiled in the market square",
    "an iron statue was unveiled in the castle courtyard",
    "a crystal statue was unveiled in the palace hall",
    "a wooden statue was unveiled in the forest clearing",
    "a stone statue was unveiled in the riverside park",
    "a copper statue was unveiled in the guild district",
]


def _generate_code():
    """Generate a random secret code like XJ-7392."""
    return f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}-{random.randint(1000, 9999)}"


def _make_timeline(num_events: int, target_event: str, target_code: str, ratio: float) -> str:
    """
    Build a timeline where the target is one of many statue-unveiling events.
    The model must distinguish the specific target (golden statue, central square)
    from similar statue-unveiling distractors to extract the code.
    """
    # Fill with generic non-statue events
    events = random.sample(EVENTS_POOL, min(num_events - len(STATUE_EVENTS) - 1, len(EVENTS_POOL)))
    while len(events) < num_events - len(STATUE_EVENTS) - 1:
        events.append("the people gathered for a ceremony")

    # Add all statue events (including target) so target is NOT lexically unique
    all_statue_events = STATUE_EVENTS + [target_event]
    random.shuffle(all_statue_events)
    events.extend(all_statue_events)

    # Shuffle everything
    random.shuffle(events)

    # Place target at desired position by replacing the event at that index
    idx = int(ratio * len(events))
    # Remove target if it happens to be elsewhere, then insert at idx
    events = [e for e in events if e != target_event]
    events.insert(idx, target_event)

    return "\n".join(f"Year {1000 + i}: {e}." for i, e in enumerate(events))


def run_temporal_narrative(
    model_name: str,
    num_events: int,
    num_examples: int,
    out_dir: str,
    depths: List[float] = None,
) -> Dict[str, Any]:
    """Run temporal narrative experiment with embedded similar distractors."""
    ensure_dir(out_dir)

    if depths is None:
        depths = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

    results = {}
    start = time.time()

    for depth in depths:
        logger.info(f"[NARRATIVE] Depth {depth:.1%}")
        preds = []
        for i in tqdm(range(num_examples), desc=f"Narrative {depth:.1%}", leave=False):
            target_code = _generate_code()
            target_event = f"a golden statue was unveiled in the central square (CODE: {target_code})"
            timeline = _make_timeline(num_events, target_event, target_code, depth)

            prompt = (
                f"Read the following timeline of historical events carefully.\n\n"
                f"{timeline}\n\n"
                f"Question: What is the secret code for the golden statue that was unveiled "
                f"in the central square? Answer with only the code."
            )
            ans = generate_text(
                [{"role": "user", "content": prompt}],
                model_name=model_name,
                max_new_tokens=15,
            )
            correct = exact_match_score(ans, target_code)
            expected_year = 1000 + int(depth * num_events)
            preds.append({
                "model_answer": ans,
                "correct": correct,
                "expected_year": expected_year,
                "target_code": target_code,
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
