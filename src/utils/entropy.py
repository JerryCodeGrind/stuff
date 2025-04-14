import numpy as np
from functools import lru_cache
from typing import Dict, Tuple

@lru_cache(maxsize=128)
def calculate_entropy(probabilities_tuple: tuple) -> float:
    """Calculate the entropy of a probability distribution."""
    # Convert tuple back to dictionary for calculation
    probabilities = dict(probabilities_tuple)
    return -sum(p * np.log2(p) for p in probabilities.values() if p > 0)

def has_confident_diagnosis(probabilities: Dict[str, float], threshold: float = 0.75) -> bool:
    """Check if any diagnosis has a probability above the threshold."""
    return any(prob >= threshold for prob in probabilities.values()) 