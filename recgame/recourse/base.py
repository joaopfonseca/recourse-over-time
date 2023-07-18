from abc import ABC, abstractmethod
from typing import Union
from copy import deepcopy
from sklearn.base import BaseEstimator
import numpy as np
from ._action_set import ActionSet


class BaseRecourse(ABC, BaseEstimator):
    """
    Base class to define recourse methods.
    """

    _estimator_type = "recourse"

    def __init__(
        self,
        model,
        threshold=0.5,
        categorical: Union[list, np.ndarray] = None,
        immutable: Union[list, np.ndarray] = None,
        step_direction: dict = None,
        y_desired: Union[int, str] = 1,
    ):
        self.model = model
        self.threshold = threshold
        self.categorical = categorical
        self.immutable = immutable
        self.step_direction = step_direction
        self.y_desired = y_desired

    def _get_coefficients(self):
        """Utility function to retrieve model parameters."""

        model = deepcopy(self.model)
        intercept = self.model.intercept_
        coefficients = self.model.coef_

        # Adjusting the intercept to match the desired threshold.
        intercept = intercept - np.log(self.threshold / (1 - self.threshold))
        model.intercept_ = intercept

        return intercept, coefficients, model

    @abstractmethod
    def _counterfactual(self, agent, action_set):
        pass

    def counterfactual(self, X, action_set=None):
        """TODO: Add documentation"""

        if not hasattr(self, "action_set_"):
            self.set_actions(X=X, action_set=action_set)

        counterfactual_examples = X.apply(
            lambda agent: self._counterfactual(agent, self.action_set_), axis=1
        )

        return counterfactual_examples

    def set_actions(self, X, action_set=None):
        """
        To be configured with the ActionSet object from the
        ``actionable-recourse`` library.
        """

        categorical = [] if self.categorical is None else self.categorical
        immutable = [] if self.immutable is None else self.immutable
        step = {} if self.step_direction is None else self.step_direction

        if action_set is None:
            action_set = ActionSet(X=X, y_desired=self.y_desired, default_bounds=(0, 1))
            for col in X.columns:
                if col in immutable:
                    action_set[col].actionable = False

                if col in step.keys():
                    action_set[col].step_direction = self.step_direction[col]

                if col in categorical:
                    action_set[col].variable_type = int
                else:
                    action_set[col].ub = X[col].max()
                    action_set[col].lb = X[col].min()

        self.action_set_ = action_set

        return self
