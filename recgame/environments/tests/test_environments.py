import pytest
from itertools import product
import numpy as np
from sklearn.linear_model import LogisticRegression
from recgame.environments import BaseEnvironment
from recgame.utils import generate_synthetic_data
from recgame.utils._testing import all_environments
from recgame.recourse import NFeatureRecourse
from recgame.environments._behavior_functions import BEHAVIOR_FUNCTIONS

RANDOM_SEED = 42


def data_source_func(n_agents):
    return generate_synthetic_data(
        n_agents=n_agents, n_continuous=3, n_categorical=0  # , random_state=RANDOM_SEED
    )[0]


df, y, _ = generate_synthetic_data(
    n_agents=100, n_continuous=3, n_categorical=0, random_state=RANDOM_SEED
)


@pytest.mark.parametrize("name, Environment", all_environments())
def test_environments(name, Environment):
    """
    Test general parameters in the different environments.
    """
    model = LogisticRegression(random_state=RANDOM_SEED).fit(df, y)
    rec = NFeatureRecourse(
        model=model,
    )

    if name == "BaseEnvironment":
        kwargs = {
            "threshold": 0.8,
            "threshold_type": "relative",
            "adaptation": 0.3,
            "behavior_function": "continuous_flexible",
            "growth_rate": 0.2,
            "growth_rate_type": "relative",
        }
    else:
        raise TypeError("Environment paramenters undefined.")

    env = Environment(
        X=df,
        recourse=rec,
        data_source_func=data_source_func,
        random_state=RANDOM_SEED,
        **kwargs
    )

    assert env.step_ == 0
    env.simulate(6)

    assert env.X_.shape[0] == 100
    assert env.step_ == 6
    assert (
        np.array([meta["threshold"] for meta in env.metadata_.values()]) >= 0.5
    ).all()


@pytest.mark.parametrize(
    "threshold, growth_rate, behavior_function",
    product([0, 5, 10, 15], [0, 5, 10, 15], BEHAVIOR_FUNCTIONS.keys()),
)
def test_absolute_params(threshold, growth_rate, behavior_function):
    """Test absolute values for growth rate and threshold."""
    model = LogisticRegression(random_state=RANDOM_SEED).fit(df, y)
    rec = NFeatureRecourse(model=model)
    env = BaseEnvironment(
        X=df,
        recourse=rec,
        data_source_func=data_source_func,
        threshold=threshold,
        threshold_type="absolute",
        growth_rate=growth_rate,
        growth_rate_type="absolute",
        adaptation=0.1,
        behavior_function=behavior_function,
        remove_winners=True,
        random_state=RANDOM_SEED,
    )
    env.simulate(3)

    assert env.X_.shape[0] == (df.shape[0] + env.step_ * (growth_rate - threshold))


# def test_remove_winners():
#     """Test if remove winners parameter is working as intended."""
