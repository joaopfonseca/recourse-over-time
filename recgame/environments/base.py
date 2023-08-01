from abc import ABC
from typing import Union
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ._behavior_functions import BEHAVIOR_FUNCTIONS
from ._analysis import EnvironmentAnalysis
from ..visualization import EnvironmentPlot


class BaseEnvironment(ABC, BaseEstimator):
    """
    Define the constraints in the environment. The central piece of the
    multi-agent analysis of algorithmic recourse.

    ``random_state`` is only relevant for visualizations or if
    ``adaptation_type in ['binary', gaussian]``. For a closed environment (no new
    agents) define ``growth_rate = 0``.

    Parameters
    ----------
    - population: ``Population`` object containing the Agents' information

    - recourse: Recourse method that allows the production of counterfactuals

    - adaptation: Global adaptation rate.
        **OUTDATED** If adaptation_type is stepwise, it represents the rate of adaptation
        of agents towards their counterfactuals. If adaptation_type is binary, the
        adaptation value determines the likelihood of adaptation.  If adaptation_type is
        binary_fixed, the adaptation value determines the number of adaptations.  If
        adaptation_type is gaussian, the adaptation value determines the flexibility of
        agents to adapt.

    - adaptation_type: can be binary, binary_fixed, stepwise, or gaussian

    - threshold: probability threshold to consider an agent's outcome as
      favorable/unfavorable

    - threshold_type: can be fixed, relative or absolute

    - growth_rate_type: can be absolute or relative

    - remove_winners: can be True, False
    """

    _estimator_type = "environment"

    def __init__(
        self,
        X,
        recourse,
        data_source_func,
        threshold: Union[float, int, list, np.ndarray] = 0.5,
        threshold_type: str = "absolute",  # "fixed", "relative", "absolute"
        adaptation: Union[float, int, list, np.ndarray] = 1.0,
        behavior_function: str = "continuous_flexible",
        # binary_constant, binary_flexible, continuous_constant, continuous_flexible
        growth_rate: Union[float, int, list, np.ndarray] = 1.0,
        growth_rate_type: str = "relative",  # "absolute", "relative"
        remove_winners: bool = True,
        random_state: int = None,
    ):
        self.X = X
        self.recourse = recourse
        self.data_source_func = data_source_func
        self.threshold = threshold
        self.threshold_type = threshold_type
        self.adaptation = adaptation
        self.behavior_function = behavior_function
        self.growth_rate = growth_rate
        self.growth_rate_type = growth_rate_type
        self.remove_winners = remove_winners
        self.random_state = random_state

        self.plot = EnvironmentPlot(self, random_state=random_state)
        self.analysis = EnvironmentAnalysis(self)
        self._check()
        self._save_metadata()

    def _check(self):
        """
        Set up variables to monitor the environment over time steps.
        """
        if not hasattr(self, "X_"):
            self.X_ = deepcopy(self.X)

        if not hasattr(self, "step_"):
            self.step_ = 0

        if not hasattr(self.recourse, "action_set_"):
            self.recourse.set_actions(self.X_)

        if not hasattr(self, "model_"):
            self.model_ = deepcopy(self.recourse.model)

        if not hasattr(self, "growth_k_"):
            self.growth_k_ = self.get_number_new_agents()

        if not hasattr(self, "threshold_"):
            self.threshold_ = self.get_score_threshold()

        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(self.random_state)

        if not hasattr(self, "outcome_"):
            self.outcome_, self.scores_ = self.outcome(return_scores=True)

        if (
            not hasattr(self, "behavior_function_")
            and type(self.behavior_function) == str
        ):
            self.behavior_function_ = BEHAVIOR_FUNCTIONS[self.behavior_function](self)
        else:
            self.behavior_function_ = self.behavior_function(self)

        if not hasattr(self, "effort_"):
            self.effort_ = self.behavior_function_.effort(
                X=self.X_, global_adaptation=self.get_global_adaptation()
            )

        if not hasattr(self, "_max_id"):
            self._max_id = self.X_.index.max()

    def _save_metadata(self):
        """
        Store metadata for each step.
        """

        if not hasattr(self, "metadata_"):
            self.metadata_ = {}

        self.metadata_[self.step_] = {
            "X": deepcopy(self.X_),
            "effort": deepcopy(self.effort_),
            "outcome": self.outcome_,
            "score": self.scores_,
            "threshold": self.threshold_,
            "growth_k": self.growth_k_,
            "threshold_index": self.threshold_index_,
            "model": deepcopy(self.model_),
        }

    def _get_moving_agents(self, step):
        """Get indices of agents that adapted between ``step-1`` and ``step``."""
        if step == 0:
            raise IndexError("Agents cannot move at the initial state (``step=0``).")

        # score = self.metadata_[step - 1]["model"].predict_proba(
        #     env.metadata_[step]["X"]
        # )[:, 1]

        adapted = (
            (self.metadata_[step - 1]["effort"] > 0)
            & (~self.metadata_[step - 1]["outcome"].astype(bool))
            & (
                self.metadata_[step - 1]["score"]
                < self.metadata_[step - 1]["threshold"]
            )
        )
        return adapted[adapted].index.values

    def set_params(self, **params):
        """
        Overwrite set_params function from scikit-learn to also update parameters passed
        in _check.

        NOTE: INCOMPLETE
        """
        super(self).set_params(**params)

        return self

    def get_global_adaptation(self, step=None):
        if step is None:
            step = self.step_

        adaptation = (
            self.adaptation[step]
            if type(self.adaptation) == [np.ndarray, list]
            else self.adaptation
        )
        return adaptation

    def get_number_new_agents(self, step=None):
        """
        Returns the number of agents to add at the current time step.

        NOTE: Formerly _update_growth_rate
        """
        if step is not None:
            return self.metadata_[step]["growth_k"]

        growth_rate = (
            self.growth_rate[self.step_]
            if type(self.growth_rate) == [np.ndarray, list]
            else self.growth_rate
        )

        if self.growth_rate_type == "absolute":
            growth_k = int(np.round(growth_rate))
        elif self.growth_rate_type == "relative":
            growth_k = int(np.round(self.X_.shape[0] * growth_rate))
        else:
            raise KeyError

        return growth_k

    def get_score_threshold(self, step=None):
        """
        Gets the score threshold ($s_t$).

        NOTE: Formerly _update_threshold
        """
        if step is not None:
            return self.metadata_[step]["threshold"]

        threshold = (
            self.threshold[self.step_]
            if type(self.threshold) in [np.ndarray, list]
            else self.threshold
        )

        if self.threshold_type == "relative":
            self.threshold_index_ = int(np.round(threshold * self.X_.shape[0]))
        elif self.threshold_type == "absolute":
            self.threshold_index_ = self.X_.shape[0] - threshold
        elif self.threshold_type == "fixed":
            probabilities = self.model_.predict_proba(self.X_)[:, 1]
            self.threshold_index_ = (probabilities <= threshold).sum()
            return threshold
        else:
            raise NotImplementedError()

        if self.threshold_index_ < 0:
            self.threshold_index_ = self.X_.shape[0] - 1

        probabilities = self.model_.predict_proba(self.X_)[:, 1]
        threshold_ = (
            probabilities[np.argsort(probabilities)][self.threshold_index_]
            if threshold != 0
            else np.nan
        )

        return threshold_

    def get_all_agents(self, keep="last"):
        """
        Retrieve a pandas dataframe with all agents that ever entered the environment.
        """
        agents = pd.concat(
            [self.metadata_[step]["X"] for step in self.metadata_.keys()]
        )
        agents = agents[~agents.index.duplicated(keep=keep)].sort_index()
        return agents

    def remove_agents(self, mask=None):
        """
        Remove agents passed in ``mask``. If None, removes all agents with a currently
        favorable outcome.
        """
        if mask is None:
            mask = self.outcome_

        indices = np.where(~mask.astype(bool))[0]
        self.X_ = self.X_.iloc[indices]
        self.effort_ = self.effort_.iloc[indices]
        return self

    def counterfactual(self, X=None, step=None):
        """
        Generates the counterfactuals over the population.
        """
        if X is None:
            X = self.X_ if step is None else self.metadata_[step]["X"]

        # Ensure algorithmic recourse parameters are up-to-date
        self.recourse.threshold = self.threshold_
        self.recourse.model = self.model_

        # Ensure model and threshold are up-to-date
        threshold = (
            self.threshold_ if step is None else self.metadata_[step]["threshold"]
        )
        model = self.model_ if step is None else self.metadata_[step]["model"]

        if np.isnan(threshold):
            return X
        else:
            # Set up recourse with temporarily modified parameters for the cases where
            # step != None
            rec = deepcopy(self.recourse)
            rec.threshold = threshold
            rec.model = model
            return rec.counterfactual(X)

    def outcome(self, X=None, step=None, return_scores=False):
        """
        Produces the outcomes for a single agent or a population.

        If ``return_scores=True``, the function will return both the outcome and scores.

        NOTE: Formerly predict
        """
        # if step is None:
        #     step = self.step_

        threshold_idx = (
            self.metadata_[step]["threshold_index"]
            if step is not None
            else self.threshold_index_
        )

        model = self.metadata_[step]["model"] if step is not None else self.model_

        if X is None:
            X = self.metadata_[step]["X"] if step is not None else self.X_

        # Output should consider threshold and return a single value per
        # observation
        probs = model.predict_proba(X)[:, 1]

        pred = np.zeros(probs.shape, dtype=int)
        idx = np.argsort(probs)[threshold_idx:]
        pred[idx] = 1

        if return_scores:
            return (
                pd.Series(pred, index=X.index),
                pd.Series(probs, index=X.index),
            )
        else:
            return pd.Series(pred, index=X.index)

    def simulate(self, steps=1):
        """
        NOTE: Formerly run_simulation
        """
        for _ in range(steps):
            self.step_ += 1
            self._simulate()
            self._save_metadata()
        return self

    def _simulate(self):
        """
        Moves environment to the next time step.

        Updates threshold, agents in population, model (?)
        The update must:
        - Remove the Agents with favorable outcomes
        - Add new Agents
        - Remove Agents with adverse outcomes that would give up

        NOTE: Formerly update
        """

        # Remove agents with favorable outcome from previous step
        if self.remove_winners:
            self.remove_agents()

        # Adapt agents
        # NOTE:
        # - Recourse will use the model from the previous time step (they adapt according
        #   to the recommendations received using the state of the previous time step)
        counterfactuals = self.counterfactual(X=self.X_, step=self.step_ - 1)
        new_factuals = self.behavior_function_.adaptation(
            factuals=self.X_, counterfactuals=counterfactuals, effort_rate=self.effort_
        )
        self.X_ = new_factuals

        # Add new agents
        self._new_agents = self.data_source_func(self.growth_k_)
        self._new_agents.index = range(
            self._max_id + 1, self._max_id + self._new_agents.shape[0] + 1
        )
        self.X_ = pd.concat([self.X_, self._new_agents])
        self._max_id = self.X_.index.max()

        # Update variables
        self.growth_k_ = self.get_number_new_agents()
        self.threshold_ = self.get_score_threshold()
        self.outcome_, self.scores_ = self.outcome(return_scores=True)
        self.effort_ = self.behavior_function_.effort(
            X=self.X_, global_adaptation=self.get_global_adaptation()
        )
        self.behavior_function_.environment = self

        return self
