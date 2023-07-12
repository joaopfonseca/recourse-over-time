import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from recgame.utils import generate_synthetic_data
from recgame.utils._testing import all_environments
from recgame.recourse import NFeatureRecourse

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


# def test_shrinking_population():
#     """Test if Environments are able to handle population sizes moving towards zero."""


# def test_remove_winners():
#     """Test if remove winners parameter is working as intended."""
