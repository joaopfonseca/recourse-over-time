import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from recgame.environments import BaseEnvironment
from recgame.recourse import NFeatureRecourse
from recgame.utils import generate_synthetic_data
from recgame.environments._behavior_functions import BEHAVIOR_FUNCTIONS


RANDOM_SEED = 42


def data_source_func(n_agents):
    return generate_synthetic_data(
        n_agents=n_agents,
        n_continuous=3,
        n_categorical=0
        # , random_state=RANDOM_SEED
    )[0]


df, y, _ = generate_synthetic_data(
    n_agents=100, n_continuous=3, n_categorical=0, random_state=RANDOM_SEED
)

model = LogisticRegression(random_state=RANDOM_SEED).fit(df, y)


@pytest.mark.parametrize("behavior_func_string", BEHAVIOR_FUNCTIONS.keys())
def test_behavior_func_environment_string(behavior_func_string):
    env = BaseEnvironment(
        X=df,
        recourse=NFeatureRecourse(model),
        data_source_func=data_source_func,
        threshold=10,
        adaptation=0.5,
        behavior_function=behavior_func_string,
        growth_rate=10,
        growth_rate_type="absolute",
        random_state=RANDOM_SEED,
    )
    env.simulate(3)


@pytest.mark.parametrize("behavior_func_obj", BEHAVIOR_FUNCTIONS.values())
def test_behavior_func_environment_obj(behavior_func_obj):
    env = BaseEnvironment(
        X=df,
        recourse=NFeatureRecourse(model),
        data_source_func=data_source_func,
        threshold=10,
        adaptation=0.5,
        behavior_function=behavior_func_obj,
        growth_rate=10,
        growth_rate_type="absolute",
        random_state=RANDOM_SEED,
    )
    env.simulate(3)


@pytest.mark.parametrize("behavior_func_obj", BEHAVIOR_FUNCTIONS.values())
def test_data_types(behavior_func_obj):
    X = df.copy()
    X["f_2"] = X["f_2"].round().astype(int)

    env = BaseEnvironment(
        X=X,
        recourse=NFeatureRecourse(model),
        data_source_func=data_source_func,
        threshold=10,
        adaptation=0.5,
        behavior_function=behavior_func_obj,
        growth_rate=10,
        growth_rate_type="absolute",
        random_state=RANDOM_SEED,
    )
    factuals = pd.DataFrame(np.random.random((30, 3)), columns=X.columns)
    counterfactuals = pd.DataFrame(np.random.random((30, 3)), columns=X.columns)
    effort_rate = pd.Series(np.random.random(30))

    new_agents = behavior_func_obj(env).adaptation(
        factuals, counterfactuals, effort_rate
    )
    assert (new_agents.drop(columns="f_2").dtypes == X.drop(columns="f_2").dtypes).all()
