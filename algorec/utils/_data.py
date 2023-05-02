import numpy as np
import pandas as pd


def generate_synthetic_data(n_agents, n_continuous, n_categorical, random_state):
    """Generate synthetic data with normal distribution."""
    continuous = [f"f_{i}" for i in range(n_continuous)]
    categorical = [f"cat_{i}" for i in range(n_categorical)]
    rng = np.random.default_rng(random_state)
    X = pd.DataFrame(
        rng.normal(loc=0.5, scale=1 / 3, size=(n_agents, n_continuous)),
        columns=continuous,
    )

    for cat in categorical:
        X[cat] = rng.integers(0, 2, n_agents)

    y = rng.integers(0, 2, n_agents)

    return X, y, categorical
