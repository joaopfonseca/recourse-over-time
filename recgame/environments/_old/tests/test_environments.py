import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from recgame.utils import generate_synthetic_data
from recgame.utils._testing import all_environments
from recgame.recourse import NFeatureRecourse

RANDOM_SEED = 42
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

    if name == "BankLoanApplication1":
        kwargs = {"threshold": 0.8, "adaptation": 0.3, "growth_rate": 0.2}
    elif name == "BankLoanApplication2":
        kwargs = {"threshold": 10, "adaptation": 15, "growth_rate": 10}
    elif name in ["BankLoanApplication3", "WillingnessEnvironment"]:
        kwargs = {"threshold": 10, "adaptation": 0.5, "growth_rate": 10}
    else:
        raise TypeError("Environment paramenters undefined.")

    env = Environment(
        X=df,
        recourse=rec,
        # n_favorable=10,  # TODO: uncomment this later on
        random_state=RANDOM_SEED,
        **kwargs
    )

    assert env.step_ == 0
    env.run_simulation(6)

    assert env.X_.shape[0] == 100
    assert env.step_ == 6
    assert (
        np.array([meta["threshold"] for meta in env.metadata_.values()]) >= 0.5
    ).all()


# def test_shrinking_population():
#     """Test if Environments are able to handle population sizes moving towards zero."""


# def test_remove_winners():
#     """Test if remove winners parameter is working as intended."""
