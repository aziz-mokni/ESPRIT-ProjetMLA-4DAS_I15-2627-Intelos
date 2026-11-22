"""Load config.yaml with paths relative to the project root (works on any OS)."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=ROOT / "configs" / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
