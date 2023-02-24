import numpy as np
from .base import BaseRecourse


class NFeatureRecourse(BaseRecourse):
    def __init__(self, model, n_features=None, threshold=0.5):
        self.model = model
        self.n_features = n_features
        self.threshold = threshold

    def _counterfactual(self, agent, action_set):
        intercept, coefficients, model = self._get_coefficients()

        # Do not change if the agent is over the threshold
        if self.model.predict_proba(agent.to_frame().T)[0, -1] >= self.threshold:
            return agent

        # Get base vector
        base_vector = coefficients.copy().squeeze()
        n_features = (
            base_vector.shape[0] if self.n_features is None else self.n_features
        )

        is_usable = np.array(
            [
                action_set[col].step_direction in [np.sign(coeff), 0]
                and action_set[col].actionable
                for col, coeff in zip(agent.index, base_vector)
            ]
        )
        base_vector[~is_usable] = 0

        # Use features with highest contribution towards the threshold
        rejected_features = np.argsort(np.abs(base_vector))[:-n_features]
        base_vector[rejected_features] = 0

        base_vector = base_vector / np.linalg.norm(base_vector)
        reg_target = -intercept
        multiplier = (reg_target - np.dot(agent.values, coefficients.T)) / np.dot(
            base_vector, coefficients.T
        )
        counterfactual = agent + multiplier * base_vector

        return counterfactual
