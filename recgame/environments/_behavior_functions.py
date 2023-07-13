from abc import abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class BaseBehavior(BaseEstimator):
    """
    Adaptation is expected to provide the output of the agent behavior.

    Effort is expected to provide each agent's flexibility (i.e., adaptation rate).
    """

    def __init__(self, environment):
        self.environment = environment

    @abstractmethod
    def adaptation(self):
        pass

    @abstractmethod
    def effort(self):
        pass

    def apply_behavior(self, factuals, counterfactuals, global_adaptation):
        """
        Applies changes to the population according to the adaptation type chosen.

        Returns the data for the adapted individuals, the original data, and the
        counterfactuals.
        """
        # Get effort rate for all agents
        effort_rate = self.effort(global_adaptation)

        # Get agents' behavior
        new_factuals = self.adaptation(factuals, counterfactuals, effort_rate)
        return new_factuals, counterfactuals


class BinaryConstant(BaseBehavior):
    """
    Binary adaptation with constant effort.
    """

    def adaptation(self, factuals, counterfactuals, effort_rate):
        """
        Applies binary adaptation.

        Returns new agents' profiles.
        """
        # Fetch environment variables
        action_set = self.environment.recourse.action_set_

        cf_vectors = counterfactuals - factuals

        # Update existing agents' feature values
        new_factuals = factuals + effort_rate.values.reshape(-1, 1) * cf_vectors
        new_factuals = np.clip(
            new_factuals,
            action_set.lb,
            action_set.ub,
        )
        return new_factuals

    def effort(self, X, global_adaptation):
        """
        Applies constant effort.

        Returns effort rate.
        """
        # Fetch environment variables
        rng = self.environment._rng

        effort_rate = rng.binomial(1, global_adaptation, X.shape[0])
        return pd.Series(effort_rate, index=X.index)


class ContinuousFlexible(BaseBehavior):
    def adaptation(self, factuals, counterfactuals, effort_rate):
        """
        Applies continuous adaptation.

        Returns new agents' profiles.
        """
        # Fetch environment variables
        model = self.environment.model_
        action_set = self.environment.recourse.action_set_
        rng = self.environment._rng

        # Gent counterfactual vectors
        cf_vectors = self._counterfactual_continuous_vectors(
            factuals, counterfactuals, effort_rate, rng, model
        )
        new_factuals = factuals + cf_vectors
        new_factuals = np.clip(
            new_factuals,
            action_set.lb,
            action_set.ub,
        )
        return new_factuals

    def _counterfactual_continuous_vectors(
        self, factuals, counterfactuals, effort_rate, rng, model
    ):
        score_change = np.abs(rng.normal(loc=0, scale=effort_rate))
        curr_scores = model.predict_proba(factuals)[:, 1]
        target_scores = np.clip(curr_scores + score_change, 0, 0.999)

        # Get base vectors
        base_vectors = counterfactuals - factuals
        base_dist = np.linalg.norm(base_vectors, axis=1).reshape(-1, 1)
        base_vectors = base_vectors / base_dist

        # Get temporary logistic regressions.
        # Counterfactuals are assumed to be perceived as the minimum requirement to
        # flip the outcome (i.e., E(y_{cf}) = 0.5). Hence, its linear transformation
        # becomes zero.
        curr_scores_lin = np.log(curr_scores / (1 - curr_scores))

        alphas = -curr_scores_lin / (base_vectors * base_vectors).sum(1)
        intercepts = curr_scores_lin - alphas * (base_vectors * factuals).sum(1)
        coefs = (
            np.repeat(np.expand_dims(alphas, axis=1), base_vectors.shape[1], axis=1)
            * base_vectors
        )

        # Get updated intercept
        intercepts = intercepts - np.log(target_scores / (1 - target_scores))

        # Get multiplier
        multiplier = (-intercepts - (factuals * coefs).sum(1).squeeze()) / (
            base_vectors * coefs
        ).sum(1).squeeze()

        cf_vectors = multiplier.values.reshape(-1, 1) * base_vectors
        cf_vectors.loc[base_dist == 0] = 0
        return cf_vectors

    def effort(self, X, global_adaptation):
        """
        Applies flexible effort.

        Returns effort rate.
        """
        # Fetch environment variables
        model = self.environment.model_
        threshold = self.environment.threshold_

        # Effort rate are the standard deviations for each agent's gaussian
        scores = model.predict_proba(X)[:, 1]
        x = threshold - scores
        effort_rate = global_adaptation / (10 * np.exp(5 * x))

        return pd.Series(effort_rate, index=X.index)


# binary_flexible

# continuous_constant

BEHAVIOR_FUNCTIONS = {
    "binary_constant": BinaryConstant,
    "continuous_flexible": ContinuousFlexible,
}
