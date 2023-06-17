from abc import ABC
import numpy as np
from sklearn.base import BaseEstimator


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

    - adaptation: If adaptation_type is stepwise, it represents the rate of adaptation
        of agents towards their counterfactuals. If adaptation_type is binary, the
        adaptation value determines the likelihood of adaptation.  If adaptation_type is
        binary_fixed, the adaptation value determines the number of adaptations.  If
        adaptation_type is gaussian, the adaptation value determines the flexibility of
        agents to adapt.

    - adaptation_type: can be binary, binary_fixed, stepwise, or gaussian

    - threshold: probability threshold to consider an agent's outcome as
      favorable/unfavorable

    - threshold_type: can be fixed, dynamic or absolute

    - growth_rate_type: can be absolute or relative

    - remove_winners: can be True, False

    Attributes
    ----------
    - Current step
    -
    """

    _estimator_type = "environment"

    def __init__(
        self,
        X,
        recourse,
        threshold: Union[float, int, list, np.ndarray] = 0.5,
        threshold_type: str = "fixed",
        adaptation: Union[float, int, list, np.ndarray] = 1.0,
        adaptation_type: str = "stepwise",
        growth_rate: Union[float, int, list, np.ndarray] = 1.0,
        growth_rate_type: str = "relative",
        remove_winners: bool = True,
        random_state=None,
    ):
        self.X = X
        self.recourse = recourse
        # self.threshold = threshold
        # self.threshold_type = threshold_type
        # self.adaptation = adaptation
        # self.adaptation_type = adaptation_type
        # self.growth_rate = growth_rate
        # self.growth_rate_type = growth_rate_type
        self.remove_winners = remove_winners
        self.random_state = random_state

        self.plot = EnvironmentPlot(self, random_state=random_state)
        self._check()
        self._save_metadata()

    def _check(self):
        """
        Set up variables to monitor the environment over time steps.
        """
        # the original parameters must be kept unchanged
        if not hasattr(self, "X_"):
            self.X_ = deepcopy(self.X)

        if not hasattr(self, "step_"):
            self.step_ = 0

        if not hasattr(self.recourse, "action_set_"):
            self.recourse.set_actions(self.X_)

        if not hasattr(self, "model_"):
            self.model_ = deepcopy(self.recourse.model)

        if not hasattr(self, "growth_k_"):
            self.growth_k_ = self._update_growth_rate()

        if not hasattr(self, "threshold_"):
            self.threshold_ = self._update_threshold()

        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(self.random_state)

        if not hasattr(self, "outcome_"):
            self.outcome_, self.scores_ = self.predict(return_scores=True)

        if not hasattr(self, "adaptation_"):
            self.adaptation_ = self._update_adaptation()

        if not hasattr(self, "_max_id"):
            self._max_id = self.X_.index.max()

    def get_score_threshold(self):
        """
        Gets the current score threshold ($s_t$).

        NOTE: Formerly _update_threshold
        """
        pass

    def apply_actions(self):
        """
        Applies the adaptation/effort configuration over the population.

        NOTE: Formerly _update_adaptation
        """
        pass

    def counterfactual(self, X):
        """
        Generates the counterfactuals over the population.
        """
        pass


class EnvironmentAnalysis:
    def __init__(self, environment):
        pass
