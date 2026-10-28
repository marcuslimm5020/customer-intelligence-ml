import numpy as np
import pandas as pd

from src.monitoring import population_stability_index


def test_population_stability_index_is_near_zero_for_same_distribution():
    reference = pd.Series(np.linspace(0, 1, 1000))
    current = reference.copy()

    assert population_stability_index(reference, current) < 1e-6


def test_population_stability_index_increases_for_shifted_distribution():
    reference = pd.Series(np.linspace(0, 1, 1000))
    shifted = pd.Series(np.linspace(2, 3, 1000))

    assert population_stability_index(reference, shifted) >= 0.2
