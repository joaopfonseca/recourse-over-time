import pandas as pd
from .base import BaseEnvironment


class BankLoanApplication(BaseEnvironment):
    def __init__(
        self,
        population,
        recourse,
        threshold=0.8,
        adaptation=0.2,
        growth_rate=1.0,
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

        # Add new agents to population
        self.population_.data = pd.concat([self.population_.data, new_agents])

        self._max_id = self.population_.data.index.max()
        return self
