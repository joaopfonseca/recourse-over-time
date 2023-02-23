from abc import ABC, abstractmethod
from typing import Union
from copy import deepcopy
import numpy as np
import pandas as pd


class BaseEnvironment(ABC):
    """
    Define the constraints in the environment. The central piece of the
    multi-agent analysis of algorithmic recourse.

    Some relevant parameters will include:
    - [x] Population
    - [x] Decision model (not necessary)
    - [x] Threshold definition (should accept fixed or dynamic thresholds)
        * Could be defined as a function?
    - [x] Algorithmic Recourse method
    - [x] Distribution of Agents' percentage of adaptation
    - [] Define how this percentage changes over time:
        * Increase rate: over time people get more motivated to move towards
          counterfactual
        * Decrease rate: over time people lose motivation to move towards
          counterfactual until they eventually give up
        * Mixed: Some people get more motivation, others lose it.
    - [] Add verbosity

    Parameters:
    - population: ``Population`` object containing the Agents' information
    - recourse: Recourse method that allows the production of counterfactuals
    - adaptation: Rate of adaptation of Agents towards their counterfactuals
    - threshold: probability threshold to consider an agent's outcome as
      favorable/unfavorable

    Attributes:
    - Current step
    -
    """

    def __init__(
        self,
        population,
        recourse,
        threshold=0.5,
        adaptation: Union[float, np.ndarray, pd.Series] = 1.0,
        growth_rate=0.0,
    ):
        self.population = population
        self.recourse = recourse
        self.threshold = threshold
        self.adaptation = adaptation
        self.growth_rate = growth_rate

        self._check()
        self.save_metadata()

    def _check(self):
        # the original population parameter should not be modified
        if not hasattr(self, "population_"):
            self.population_ = deepcopy(self.population)

        if not hasattr(self, "step_"):
            self.step_ = 0

        if not hasattr(self, "adaptation_"):
            if type(self.adaptation) in [int, float]:
                self.adaptation_ = pd.Series(
                    self.adaptation, index=self.population_.data.index
                )
            else:
                raise NotImplementedError()

        if not hasattr(self, "model_"):
            self.model_ = deepcopy(self.recourse.model)

        return self

    def predict(self, population=None):
        """
        Produces the outcomes for a single agent or a population.
        """

        if population is None:
            population = self.population_

        # Output should consider threshold and return a single value per
        # observation
        return (
            self.model_.predict_proba(population.data)[:, 1] >= self.threshold
        ).astype(int)

    def counterfactual(self, population=None):
        """
        Get the counterfactual examples.
        """
        if population is None:
            population = self.population_
        return self.recourse.counterfactual(population)

    def _counterfactual_vectors(self, factuals, counterfactuals):
        # Compute avg distance to counterfactual
        base_dist = np.linalg.norm(counterfactuals - factuals, axis=1).reshape(-1, 1)
        avg_dist = base_dist[~np.isnan(base_dist)].mean()

        # Get base counterfactual vectors
        cf_vectors = avg_dist * ((counterfactuals - factuals) / base_dist)

        # Compute adaptation ratio
        # Compute euclidean distance for each agent
        return cf_vectors

    # @abstractmethod
    def add_agents(self, n_agents):
        pass

    def remove_agents(self, indices):
        self.population_.data = self.population_.data.iloc[indices]
        self.adaptation_ = self.adaptation_.iloc[indices]
        return self

    def save_metadata(self):
        """
        Store metadata for each step. Currently incomplete.
        """

        if not hasattr(self, "metadata_"):
            self.metadata_ = {}

        self.metadata_[self.step_] = {
            "population": deepcopy(self.population_),
            "adaptation": deepcopy(self.adaptation_),
        }

    def update(self):
        """
        Moves environment to the next time step.

        Updates threshold, agents in population, model (?)
        The update must:
        - Remove the Agents with favorable outcomes
        - Add new Agents
        - Remove Agents with adverse outcomes that would give up
        """

        # Save number of agents in population
        n_agents = self.population_.data.shape[0]

        # Remove agents with favorable outcome from previous step
        outcome = self.predict()
        indices = np.where(~outcome.astype(bool))[0]
        self.remove_agents(indices)

        # Get factuals + counterfactuals
        factuals = self.population_.data
        counterfactuals = self.counterfactual()

        # Get adaptation rate for all agents
        # This part is tricky; How can this be achieved properly?
        cf_vectors = self._counterfactual_vectors(factuals, counterfactuals)

        # Update existing agents' feature values
        new_factuals = factuals + self.adaptation_.values.reshape(-1, 1) * cf_vectors
        self.population_.data = new_factuals

        # Add new agents (replace removed agents and add new agents according
        # to growth rate)
        n_new_agents = np.round(n_agents * self.growth_rate + indices.shape[0])
        self.add_agents(n_new_agents)

        # Update metadata and step number
        self.step_ += 1
        self.save_metadata()

        return self

    def run_simulation(self, n_steps):
        """
        Simulates ``n`` runs through the environment.
        """
        for i in range(n_steps):
            self.update()
        return self
