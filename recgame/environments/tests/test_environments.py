import pytest
from itertools import product
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from recgame.environments import BaseEnvironment
from recgame.utils import generate_synthetic_data
from recgame.utils._testing import all_environments
from recgame.recourse import NFeatureRecourse
from recgame.environments._behavior_functions import BEHAVIOR_FUNCTIONS

RANDOM_SEED = 42
ENV_KWARGS = {
    "threshold": 0.8,
    "threshold_type": "relative",
    "adaptation": 0.3,
    "behavior_function": "continuous_flexible",
    "growth_rate": 0.2,
    "growth_rate_type": "relative",
}


def data_source_func(n_agents):
    return generate_synthetic_data(
        n_agents=n_agents, n_continuous=3, n_categorical=0  # , random_state=RANDOM_SEED
    )[0]


def data_source_func_cat(n_agents):
    df = pd.DataFrame(np.random.random((n_agents, 3)), columns=["a", "b", "c", "d"])
    df["cat_1"] = np.random.integers(0, 2, n_agents)
    df["cat_2"] = np.random.integers(0, 2, n_agents)
    return df


df, y, _ = generate_synthetic_data(
    n_agents=100, n_continuous=3, n_categorical=0, random_state=RANDOM_SEED
)

df2, _, _ = generate_synthetic_data(
    n_agents=100, n_continuous=3, n_categorical=2, random_state=RANDOM_SEED
)


@pytest.mark.parametrize("name, Environment", all_environments())
def test_environments(name, Environment):
    """
    Test general parameters in the different environments.
    """
    model = LogisticRegression(random_state=RANDOM_SEED).fit(df, y)
    rec = NFeatureRecourse(model=model)

    if name != "BaseEnvironment":
        raise TypeError("Environment paramenters undefined.")

    env = Environment(
        X=df,
        recourse=rec,
        data_source_func=data_source_func,
        random_state=RANDOM_SEED,
        **ENV_KWARGS
    )

    assert env.step_ == 0
    env.simulate(6)

    assert env.X_.shape[0] == 100
    assert env.step_ == 6
    assert (
        np.array([meta["threshold"] for meta in env.metadata_.values()]) >= 0.5
    ).all()
    assert (env.X_.dtypes == env.X.dtypes).all()


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
    assert (env.X_.dtypes == env.X.dtypes).all()


# def test_remove_winners():
#     """Test if remove winners parameter is working as intended."""


@pytest.mark.parametrize("name, Environment", all_environments())
def test_data_types(name, Environment):
    model = LogisticRegression(random_state=RANDOM_SEED).fit(df, y)
    rec = NFeatureRecourse(model=model, categorical=["cat_1", "cat_2"])

    if name != "BaseEnvironment":
        raise TypeError("Environment paramenters undefined.")

    env = Environment(
        X=df2,
        recourse=rec,
        data_source_func=data_source_func,
        random_state=RANDOM_SEED,
        **ENV_KWARGS
    )

    env.simulate()
    assert (env.X_.dtypes == env.X.dtypes).all()
