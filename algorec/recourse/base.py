from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np


class BaseRecourse(ABC):
    """
    Base class to define recourse methods.
    """
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

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

    def counterfactual(self, population):
        action_set = population.action_set_

        counterfactual_examples = population.data.apply(
            lambda agent: self._counterfactual(agent, action_set), axis=1
        )

        return counterfactual_examples


