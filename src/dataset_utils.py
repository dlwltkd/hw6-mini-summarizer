# src/dataset_utils.py
import json
import random
from typing import List, Dict, Tuple

def load_data(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def train_test_split(
    examples: List[Dict],
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    shuffled = examples[:]  # copy
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = int(n_total * test_ratio)

    test = shuffled[:n_test]
    train = shuffled[n_test:]
    return train, test
