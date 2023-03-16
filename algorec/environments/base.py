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
    - [ ] Define how this percentage changes over time:
        * Increase rate: over time people get more motivated to move towards
          counterfactual
        * Decrease rate: over time people lose motivation to move towards
          counterfactual until they eventually give up
        * Mixed: Some people get more motivation, others lose it.
    - [ ] Add verbosity

    Parameters:
    - population: ``Population`` object containing the Agents' information
    - recourse: Recourse method that allows the production of counterfactuals
    - adaptation: Rate of adaptation of Agents towards their counterfactuals
    - adaptation_type: can be binary or stepwise
    - threshold: probability threshold to consider an agent's outcome as
      favorable/unfavorable
    - threshold_type: can be fixed, dynamic or absolute
    - growth_rate_type: can be absolute or relative
    - remove_winners: can be True, False

    Attributes:
    - Current step
    -
    """

    def __init__(
        self,
        population,
        recourse,
        threshold: Union[float, np.ndarray] = 0.5,
        threshold_type: str = "fixed",
        adaptation: Union[float, np.ndarray, pd.Series] = 1.0,
        adaptation_type: str = "stepwise",
        growth_rate: Union[float, int] = 1.0,
        growth_rate_type: str = "dynamic",
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
        self.growth_rate_type = growth_rate_type
        self.remove_winners = remove_winners
        self.random_state = random_state

        self.plot = EnvironmentPlot(self, random_state=random_state)
        self._check()
        self._save_metadata()

    def _check(self):
        # the original parameters must not be changed
        if not hasattr(self, "population_"):
            self.population_ = deepcopy(self.population)

        if not hasattr(self, "step_"):
            self.step_ = 0

        if not hasattr(self, "model_"):
            self.model_ = deepcopy(self.recourse.model)

        if not hasattr(self, "growth_k_"):
            self._update_growth_rate()

        if not hasattr(self, "threshold_"):
            self._update_threshold()

        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(self.random_state)

        if not hasattr(self, "adaptation_"):
            self._update_adaptation()

        if not hasattr(self, "outcome_"):
            self.outcome_ = self.predict()

        if not hasattr(self, "_max_id"):
            self._max_id = self.population_.data.index.max()

    def _update_threshold(self):
        threshold = (
            self.threshold[self.step_]
            if type(self.threshold) == np.ndarray
            else self.threshold
        )

        if self.threshold_type == "dynamic":
            self.threshold_index_ = int(
                np.round(threshold * self.population_.data.shape[0])
            )
            probabilities = self.model_.predict_proba(self.population_.data)[:, 1]
            self.threshold_ = probabilities[np.argsort(probabilities)][
                self.threshold_index_
            ]

        elif self.threshold_type == "absolute":
            self.threshold_index_ = self.population_.data.shape[0] - threshold
            if self.threshold_index_ < 0:
                raise KeyError("Threshold cannot be larger than the population.")
            probabilities = self.model_.predict_proba(self.population_.data)[:, 1]
            self.threshold_ = (
                probabilities[np.argsort(probabilities)][self.threshold_index_]
                if threshold != 0
                else np.nan
            )

        elif self.threshold_type == "fixed":
            self.threshold_ = threshold

        else:
            raise NotImplementedError()

        return self

    def _update_adaptation(self):
        adaptation = (
            self.adaptation[self.step_]
            if type(self.adaptation) == np.ndarray
            else self.adaptation
        )

        if self.adaptation_type == "stepwise":
            pass
        elif self.adaptation_type == "binary":
            adaptation = self._rng.binomial(
                1, adaptation, self.population_.data.shape[0]
            )

        self.adaptation_ = pd.Series(adaptation, index=self.population_.data.index)
        return self

    def _update_growth_rate(self):
        growth_rate = (
            self.growth_rate[self.step_]
            if type(self.growth_rate) == np.ndarray
            else self.growth_rate
        )

        if self.growth_rate_type == "absolute":
            self.growth_k_ = int(np.round(growth_rate))
        elif self.growth_rate_type == "relative":
            self.growth_k_ = int(np.round(self.population_.data.shape[0] * growth_rate))
        return self

    def predict(self, population=None, step=None):
        """
        Produces the outcomes for a single agent or a population.

        TODO: Refactor if statement
        """

        if step is None:
            threshold = self.threshold_
            if self.threshold_type in ["dynamic", "absolute"]:
                threshold_idx = self.threshold_index_
            if population is None:
                population = self.population_
        else:
            threshold = self.metadata_[step]["threshold"]
            if self.threshold_type in ["dynamic", "absolute"]:
                threshold_idx = self.metadata_[step]["threshold_index"]
            if population is None:
                population = self.metadata_[step]["population"]

        # Output should consider threshold and return a single value per
        # observation
        probs = self.model_.predict_proba(population.data)[:, 1]
        if self.threshold_type in ["dynamic", "absolute"]:
            pred = np.zeros(probs.shape, dtype=int)
            idx = np.argsort(probs)[threshold_idx:]
            pred[idx] = 1
        else:
            pred = (probs >= threshold).astype(int)

        return pd.Series(pred, index=population.data.index)

    def counterfactual(self, population=None, step=None):
        """
        Get the counterfactual examples.
        """
        if population is None:
            population = (
                self.population_ if step is None else self.metadata_[step]["population"]
            )

        # Ensure threshold is up-to-date
        threshold = (
            self.threshold_ if step is None else self.metadata_[step]["threshold"]
        )
        if np.isnan(threshold):
            return population.data
        else:
            self.recourse.threshold = threshold
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

    def _save_metadata(self):
        """
        Store metadata for each step. Currently incomplete.
        """

        if not hasattr(self, "metadata_"):
            self.metadata_ = {}

        self.metadata_[self.step_] = {
            "population": deepcopy(self.population_),
            "adaptation": deepcopy(self.adaptation_),
            "outcome": self.outcome_,
            "threshold": self.threshold_,
            "growth_k": self.growth_k_,
        }
        if self.threshold_type in ["dynamic", "absolute"]:
            self.metadata_[self.step_]["threshold_index"] = self.threshold_index_

    def adapt_agents(self, population):
        # Get factuals + counterfactuals
        factuals = self.population_.data
        counterfactuals = self.counterfactual()

        # Get adaptation rate for all agents
        if self.adaptation_type == "binary":
            cf_vectors = counterfactuals - factuals
        elif self.adaptation_type == "stepwise":
            cf_vectors = self._counterfactual_vectors(factuals, counterfactuals)
        else:
            raise NotImplementedError()

        # Update existing agents' feature values
        new_factuals = factuals + self.adaptation_.values.reshape(-1, 1) * cf_vectors
        return new_factuals

    def update(self):
        """
        Moves environment to the next time step.

        Updates threshold, agents in population, model (?)
        The update must:
        - Remove the Agents with favorable outcomes
        - Add new Agents
        - Remove Agents with adverse outcomes that would give up
        """

        # Remove agents with favorable outcome from previous step
        if self.remove_winners:
            self.remove_agents(self.outcome_)

        # Adapt agents
        new_factuals = self.adapt_agents(population=self.population_.data)
        self.population_.data = new_factuals

        # Add new agents
        new_agents = self.add_agents(self.growth_k_)
        new_agents.index = range(
            self._max_id + 1, self._max_id + new_agents.shape[0] + 1
        )
        self.population_.data = pd.concat([self.population_.data, new_agents])
        self._max_id = self.population_.data.index.max()

        # Update variables
        self._update_adaptation()
        self._update_threshold()
        self._update_growth_rate()

        # Determine outcome of current step
        self.outcome_ = self.predict()

        # Update metadata and step number
        self.step_ += 1
        self._save_metadata()

        return self

    def run_simulation(self, n_steps):
        """
        Simulates ``n`` runs through the environment.
        """
        for _ in range(n_steps):
            self.update()
        return self

    def _get_moving_agents(self, step):
        """Get indices of agents that adapted between ``step-1`` and ``step``."""
        if step == 0:
            raise IndexError("Agents cannot move at the initial state (``step=0``).")

        adapted = (
            (self.metadata_[step - 1]["adaptation"] > 0)
            & (~self.metadata_[step - 1]["outcome"].astype(bool))
            & (
                self.model_.predict_proba(self.metadata_[step - 1]["population"].data)[
                    :, 1
                ]
                < self.metadata_[step - 1]["threshold"]
            )
        )
        return adapted[adapted].index.values

    def success_rate(self, step, last_step=None):
        """
        For an agent to move, they need to have adaptation > 0, unfavorable outcome and
        score < threshold.

        If nan, no agents adapted (thus no rate is calculated).
        """
        if step == 0:
            raise IndexError(
                "Cannot calculate success rate at ``step=0`` (agents have not adapted "
                "yet)."
            )

        steps = [step] if last_step is None else [s for s in range(step, last_step)]
        success_rates = []
        for step in steps:
            adapted = self._get_moving_agents(step)
            outcomes = self.metadata_[step]["outcome"].astype(bool)
            favorable_outcomes = (
                self.metadata_[step]["population"].data.loc[outcomes].index
            )

            success = favorable_outcomes.intersection(adapted)
            success_rate = (
                success.shape[0] / adapted.shape[0] if adapted.shape[0] > 0 else np.nan
            )

            success_rates.append(success_rate)
        return np.array(success_rates)

    def threshold_drift(self, step, last_step=None):
        """Calculate threshold variations across"""

        if step == 0:
            raise IndexError("Cannot threshold drift at ``step=0``.")

        steps = [step] if last_step is None else [s for s in range(step, last_step)]
        steps = [step - 1] + steps
        thresholds = [self.metadata_[step]["threshold"] for step in steps]
        threshold_drift = [
            (thresholds[i] - thresholds[i - 1]) / thresholds[i - 1]
            for i in range(1, len(thresholds))
        ]
        return np.array(threshold_drift)
