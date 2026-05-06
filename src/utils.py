"""Common utilities."""
import json
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_jsonl(path: str, records: List[Dict[str, Any]]):
    """Save records as JSONL."""
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL records."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def save_json(path: str, data: Any):
    """Save data as pretty-printed JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)
