from typing import Union
import numpy as np
import pandas as pd
from .base import BaseEnvironment


class ClosedEnvironment(BaseEnvironment):
    """Closed, general-purpose environment"""

    def __init__(
        self,
        population,
        recourse,
        threshold: float = 0.5,
        threshold_type: str = "fixed",
        adaptation: Union[float, np.ndarray, pd.Series] = 1.0,
        adaptation_type: str = "stepwise",
        remove_winners: bool = True,
        random_state=None,
    ):
        super().__init__(
            population=population,
            recourse=recourse,
            threshold=threshold,
            threshold_type=threshold_type,
            adaptation=adaptation,
            adaptation_type=adaptation_type,
            growth_rate=0,
            remove_winners=remove_winners,
            random_state=random_state,
        )

    def add_agents(self, n_agents):
        pass


class BankLoanApplication1(BaseEnvironment):
    """
    Bank Loan Application Environment 1:
    - Uniformly generated data across the different features;
    - Dynamic threshold;
    - Binary adaptation;
    - Relative growth rate;
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
        all_cols = self.population.data.columns
        categorical = self.population.categorical
        categorical = (
            [] if self.population.categorical is None else self.population.categorical
        )
        continuous = all_cols.drop(categorical)

        new_agents = pd.DataFrame(
            self._rng.random((n_agents, len(continuous))),
            columns=continuous,
            index=range(self._max_id + 1, self._max_id + n_agents + 1),
        )
        for col in categorical:
            # Use the original data to retrieve the distributions
            counts = self.population.data.groupby(col).size()
            new_agents[col] = self._rng.choice(
                counts.index, size=n_agents, p=counts / counts.sum()
            ).astype(self.population_.data[col].dtype)

        return new_agents


class BankLoanApplication2(BaseEnvironment):
    """
    Bank Loan Application Environment 2:
    - Uniformly generated data across the different features;
    - Absolute threshold;
    - Gaussian adaptation;
    - Absolute growth rate;

    ``n_loans`` is the number of loans granted per time step
    ``new_agents`` is the number of new applicants entering the environment
    """

    def __init__(
        self,
        population,
        recourse,
        n_loans=10,
        adaptation=0.2,
        new_agents=10,
        random_state=None,
    ):
        super().__init__(
            population=population,
            recourse=recourse,
            threshold=n_loans,
            threshold_type="absolute",
            adaptation=adaptation,
            adaptation_type="gaussian",
            growth_rate=new_agents,
            growth_rate_type="absolute",
            remove_winners=True,
            random_state=random_state,
        )

    def add_agents(self, n_agents):
        all_cols = self.population.data.columns
        categorical = self.population.categorical
        categorical = (
            [] if self.population.categorical is None else self.population.categorical
        )
        continuous = all_cols.drop(categorical)

        new_agents = pd.DataFrame(
            self._rng.random((n_agents, len(continuous))),
            columns=continuous,
            index=range(self._max_id + 1, self._max_id + n_agents + 1),
        )
        for col in categorical:
            # Use the original data to retrieve the distributions
            counts = self.population.data.groupby(col).size()
            new_agents[col] = self._rng.choice(
                counts.index, size=n_agents, p=counts / counts.sum()
            ).astype(self.population_.data[col].dtype)

        return new_agents
