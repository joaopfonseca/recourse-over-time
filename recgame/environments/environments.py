from typing import Union
from copy import deepcopy
import numpy as np
import pandas as pd
from .base import BaseEnvironment
from ..utils import generate_synthetic_data


def _add_agents(self, n_agents):
    all_cols = self.population.X.columns
    categorical = (
        [] if self.population.categorical is None else self.population.categorical
    )
    continuous = all_cols.drop(categorical)

    X, _, _ = generate_synthetic_data(
        n_agents=n_agents,
        n_continuous=len(continuous),
        n_categorical=len(categorical),
        random_state=self._rng,
    )
    return X


class BankLoanApplication1(BaseEnvironment):
    """
    Bank Loan Application Environment 1:
    - Uniformly generated data across the different features;
    - Dynamic threshold;
    - Binary adaptation;
    - Relative growth rate;

    TODO: CHANGE PARAMETER NAMES
    """

    def __init__(
        self,
        population,
        recourse,
        threshold=0.8,
        adaptation=0.2,
        growth_rate=0.2,
        random_state=None,
    ):
        super().__init__(
            population=population,
            recourse=recourse,
            threshold=threshold,
            threshold_type="dynamic",
            adaptation=adaptation,
            adaptation_type="binary",
            growth_rate=growth_rate,
            growth_rate_type="relative",
            remove_winners=True,
            random_state=random_state,
        )

    def add_agents(self, n_agents):
        return _add_agents(self, n_agents)


class BankLoanApplication2(BaseEnvironment):
    """
    Bank Loan Application Environment 2:
    - Uniformly generated data across the different features;
    - Absolute threshold;
    - Binary_fixed adaptation;
    - Absolute growth rate;

    TODO: CHANGE PARAMETER NAMES

    NOTE: threshold used to be n_loans
    NOTE: growth_rate used to be new_agents
    ``threshold`` is the number of loans granted per time step
    ``new_agents`` is the number of new applicants entering the environment
    """

    def __init__(
        self,
        population,
        recourse,
        threshold=10,
        adaptation=20,
        growth_rate=10,
        random_state=None,
    ):
        super().__init__(
            population=population,
            recourse=recourse,
            threshold=threshold,
            threshold_type="absolute",
            adaptation=adaptation,
            adaptation_type="binary_fixed",
            growth_rate=growth_rate,
            growth_rate_type="absolute",
            remove_winners=True,
            random_state=random_state,
        )

    def add_agents(self, n_agents):
        return _add_agents(self, n_agents)


class BankLoanApplication3(BaseEnvironment):
    """
    Bank Loan Application Environment 3:
    - Uniformly generated data across the different features;
    - Absolute threshold;
    - Gaussian adaptation;
    - Absolute growth rate;

    TODO: CHANGE PARAMETER NAMES

    ``threshold`` is the number of loans granted per time step.
    ``new_agents`` is the number of new applicants entering the environment.
    ``adaptation`` adjusts the flexibility for agents to adapt.
    """

    def __init__(
        self,
        population,
        recourse,
        threshold=10,
        adaptation=0.2,
        growth_rate=10,
        random_state=None,
    ):
        super().__init__(
            population=population,
            recourse=recourse,
            threshold=threshold,
            threshold_type="absolute",
            adaptation=adaptation,
            adaptation_type="gaussian",
            growth_rate=growth_rate,
            growth_rate_type="absolute",
            remove_winners=True,
            random_state=random_state,
        )

    def add_agents(self, n_agents):
        return _add_agents(self, n_agents)


class WillingnessEnvironment(BaseEnvironment):
    """
    Adapts the gaussian adaptation to include an additional parameter ``willingness`` to
    determine how easily agents can adapt, instead of the function defined with the
    gaussian case.

    Characteristics of this environment:
    - Uniformly generated data across the different features;
    - Absolute threshold;
    - Willingness-based adaptation;
    - Absolute growth rate;

    TODO: CHANGE PARAMETER NAMES
    NOTE: threshold used to be n_loans
    NOTE: growth_rate used to be new_agents

    ``willingness`` should be a numpy vector with individual willingness values.
    """

    def __init__(
        self,
        population,
        recourse,
        threshold=10,
        adaptation=0.2,
        willingness=None,
        growth_rate=10,
        random_state=None,
    ):
        self.willingness = willingness
        super().__init__(
            population=population,
            recourse=recourse,
            threshold=threshold,
            threshold_type="absolute",
            adaptation=adaptation,
            adaptation_type="gaussian",
            growth_rate=growth_rate,
            growth_rate_type="absolute",
            remove_winners=True,
            random_state=random_state,
        )

    def _check(self):
        """Override to use willingness."""
        if self.willingness is not None and not hasattr(self, "adaptation_"):
            self.adaptation_ = deepcopy(self.willingness)
        super()._check()

    def _update_adaptation(self):
        """
        Defining individualized willingness can be based on observation, or be entirely
        arbitrary. In this setting, it is based on the initial score and it is never
        updated (i.e., arbitrary).
        """
        # This is the global adaptation, it is a float value
        adaptation = (
            self.adaptation[self.step_]
            if type(self.adaptation) == [np.ndarray, list]
            else self.adaptation
        )
        current_adaptation = self.adaptation_ if hasattr(self, "adaptation_") else None

        agents = (
            self._new_agents if hasattr(self, "_new_agents") else self.population_.X
        )

        # Define their willingness to adapt based on initial score
        # scores = self.model_.predict_proba(agents)[:, 1]
        # threshold = self.threshold_

        # # ``x`` is the invidual adaptation (willingness), ``adaptation`` is global
        # # in ``x``, with this configuration, lower is better
        # x = threshold - scores
        # adaptation = adaptation / (20 * np.exp(10 * x))
        # adaptation = pd.Series(adaptation, index=agents.index)

        # Alternative, make willingness to adapt random
        x = self._rng.random()
        adaptation = x * adaptation / 10
        adaptation = pd.Series(adaptation, index=agents.index)

        adaptation = pd.concat([current_adaptation, adaptation])

        return pd.Series(adaptation, index=self.population_.X.index)

    def add_agents(self, n_agents):
        return _add_agents(self, n_agents)
