import logging
import numpy as np
from recourse import Flipset


class ActionableRecourse:
    """
    Should work exactly as defined in the paper if the model is linear.
    Otherwise, an explainability model should be used to retrieve coefficients
    for each observation.

    NOTE: Initially only linear models will be considered

    https://arxiv.org/pdf/1809.06514.pdf
    """

    def __init__(self, model, flipset_size=100, discretize=False, sample=False):
        self.model = model
        self.flipset_size = flipset_size
        self.discretize = discretize
        self.sample = sample

    def _get_coefficients(self):
        intercept = self.model.intercept_
        coefficients = self.model.coef_
        return intercept, coefficients

    def _counterfactual(self, agent, action_set):
        factual = agent.values
        intercept, coefficients = self._get_coefficients()

        # Default counterfactual value if no action flips the prediction
        target_shape = factual.shape[0]
        empty = np.empty(target_shape)
        empty[:] = np.nan
        counterfactual = empty

        # Align action set to coefficients
        action_set.set_alignment(coefficients=coefficients)

        # Build AR flipset
        fs = Flipset(
            x=factual,
            action_set=action_set,
            coefficients=coefficients,
            intercept=intercept,
        )
        try:
            fs_pop = fs.populate(total_items=self.flipset_size)
        except (ValueError, KeyError):
            logging.warning(
                "Actionable Recourse is not able to produce a counterfactual explanation for instance {}".format(
                    agent.index
                )
            )
            logging.warning(factual)
            return counterfactual

        # Get actions to flip predictions
        actions = fs_pop.actions
        print(actions[0])

        for action in actions:
            candidate_cf = (factual + action).reshape(
                (1, -1)
            )  # Reshape to keep two-dim. input
            # Check if candidate counterfactual really flipps the prediction of ML model
            pred_cf = np.argmax(self.model.predict_proba(candidate_cf))
            pred_f = np.argmax(self.model.predict_proba(factual.reshape((1, -1))))
            if pred_cf != pred_f:
                counterfactual = candidate_cf.squeeze()
                break

        return counterfactual

    def counterfactuals(self, population):
        action_set = population._action_set
        # temporary - this is quite inefficient
        counterfactual_examples = []
        for i, agent in population.data.iterrows():
            counterfactual_examples.append(self._counterfactual(agent, action_set))

        return counterfactual_examples
