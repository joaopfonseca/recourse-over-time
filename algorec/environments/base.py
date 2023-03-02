from abc import ABC, abstractmethod
from typing import Union
from copy import deepcopy
import numpy as np
import pandas as pd

from ..visualization import EnvironmentPlot


class BaseEnvironment(ABC):
    """
    Define the constraints in the environment. The central piece of the
    multi-agent analysis of algorithmic recourse.

    ``random_state`` is only relevant for visualizations or if
    ``adaptation_type = 'binary'``. For a closed environment (no new agents) define
    ``growth_rate = 0``.

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
        threshold: float = 0.5,
        threshold_type: str = "fixed",
        adaptation: Union[float, np.ndarray, pd.Series] = 1.0,
        adaptation_type: str = "continuous",
        growth_rate: float = 1.0,
        remove_winners: bool = True,
        random_state=None,
    ):
        self.population = population
        self.recourse = recourse
        self.threshold = threshold
        self.threshold_type = threshold_type
        self.adaptation = adaptation
        self.adaptation_type = adaptation_type
        self.growth_rate = growth_rate
        self.remove_winners = remove_winners
        self.random_state = random_state

        self.plot = EnvironmentPlot(self, random_state=random_state)
        self._check()
        self.save_metadata()

    def _check(self):
        # the original parameters must should not be changed
        if not hasattr(self, "population_"):
            self.population_ = deepcopy(self.population)

        if not hasattr(self, "step_"):
            self.step_ = 0

        if not hasattr(self, "model_"):
            self.model_ = deepcopy(self.recourse.model)

        if not hasattr(self, "threshold_"):
            self._update_threshold()

        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(self.random_state)

        if not hasattr(self, "adaptation_"):
            self._update_adaptation()

        if not hasattr(self, "_max_id"):
            self._max_id = self.population_.data.index.max()

    def _update_threshold(self):
        if self.threshold_type == "dynamic":
            self.threshold_index_ = int(np.round(self.threshold * self.population_.data.shape[0]))
            probabilities = self.model_.predict_proba(self.population_.data)[:, 1]
            self.threshold_ = probabilities[np.argsort(probabilities)][self.threshold_index_]

        elif self.threshold_type == "fixed":
            self.threshold_ = self.threshold

        elif self.threshold_type not in ["fixed", "dynamic"]:
            raise NotImplementedError()

        return self

    def _update_adaptation(self):
        if self.adaptation_type == "continuous":
            adaptation = self.adaptation
        elif self.adaptation_type == "binary":
            adaptation = self._rng.binomial(
                1, self.adaptation, self.population_.data.shape[0]
            )
        else:
            raise NotImplementedError()

        self.adaptation_ = pd.Series(adaptation, index=self.population_.data.index)
        return self

    def predict(self, population=None, step=None):
        """
        Produces the outcomes for a single agent or a population.
        """

        if step is None:
            threshold = self.threshold_
            if self.threshold_type == "dynamic":
                threshold_idx = self.threshold_index_
            if population is None:
                population = self.population_
        else:
            threshold = self.metadata_[step]["threshold"]
            if self.threshold_type == "dynamic":
                threshold_idx = self.metadata_[step]["threshold_index"]
            if population is None:
                population = self.metadata_[step]["population"]

        # Output should consider threshold and return a single value per
        # observation
        probs = self.model_.predict_proba(population.data)[:, 1]
        if self.threshold_type == "dynamic":
            pred = np.zeros(probs.shape, dtype=int)
            idx = np.argsort(probs)[threshold_idx:]
            pred[idx] = 1
        else:
            pred = (probs >= threshold).astype(int)

        return pred

    def counterfactual(self, population=None):
        """
        Get the counterfactual examples.
        """
        if population is None:
            population = self.population_

        # Ensure threshold is up-to-date
        self.recourse.threshold = self.threshold_

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

    @abstractmethod
    def add_agents(self, n_agents):
        # Generate data
        # Add info to population
        pass

    def remove_agents(self, mask):
        indices = np.where(~mask.astype(bool))[0]
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
            "threshold": self.threshold_,
        }
        if self.threshold_type == "dynamic":
            self.metadata_[self.step_]["threshold_index"] = self.threshold_index_

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
        if self.remove_winners:
            outcome = self.predict()
            self.remove_agents(outcome)

        n_removed = np.sum(outcome) if self.remove_winners else 0

        # Get factuals + counterfactuals
        factuals = self.population_.data
        counterfactuals = self.counterfactual()

        # Get adaptation rate for all agents
        if self.adaptation_type == "binary":
            cf_vectors = counterfactuals - factuals
        elif self.adaptation_type == "continuous":
            cf_vectors = self._counterfactual_vectors(factuals, counterfactuals)
        else:
            raise NotImplementedError()

        # Update existing agents' feature values
        new_factuals = factuals + self.adaptation_.values.reshape(-1, 1) * cf_vectors
        self.population_.data = new_factuals

        # Add new agents (replace removed agents and add new agents according
        # to growth rate)
        if self.growth_rate != 0:
            n_new_agents = int(np.round(n_agents * (self.growth_rate - 1) + n_removed))
            self.add_agents(n_new_agents)

        # Update variables, metadata and step number
        self._update_adaptation()
        self._update_threshold()
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

    def success_rate(self, step, last_step=None):
        steps = [step] if last_step is None else [s for s in range(step, last_step)]
        success_rates = []
        for step in steps:
            adapted = self.metadata_[step]["adaptation"] > 0
            favorable_y = self.predict(step=step)
            success = favorable_y[adapted]
            success_rate = success.sum() / success.shape[0]
            print(success.sum(), success.shape[0])
            success_rates.append(success_rate)
        return np.array(success_rates)
