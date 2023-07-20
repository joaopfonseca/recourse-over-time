from abc import ABCMeta
from typing import Union
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from .base import BaseRecourse
from ..utils._utils import _optional_import


class _ThresholdClassifier(BaseEstimator, metaclass=ABCMeta):
    def __init__(self, classifier, threshold, y_desired=1):
        self.classifier = deepcopy(classifier)
        self.threshold = threshold
        self.y_desired = y_desired

    def _convert_proba(self, prob):
        """Scales a set of probabilities to ensure ``threshold`` is centered at 0.5."""
        m = 0.5 / self.threshold
        return m * prob

    def predict(self, X):
        return (self.predict_proba(X)[:, self.y_desired] >= 0.5).astype(int)

    def predict_proba(self, X):
        return self._convert_proba(self.classifier.predict_proba(X))

    def fit(self, X, y):
        return self.classifier.fit(X, y)


class DiCE(BaseRecourse):
    """
    Adapts the DiCE recourse algorithm implemented in
    https://github.com/interpretml/DiCE/tree/main to match recgame's API.
    """

    def __init__(
        self,
        model,
        model_backend="sklearn",
        threshold=0.5,
        categorical: Union[list, np.ndarray] = None,
        immutable: Union[list, np.ndarray] = None,
        step_direction: dict = None,
        y_desired: Union[int, str] = 1,
        set_size=5,
        random_state=None,
    ):
        self.model = model
        self.model_backend = model_backend
        self.threshold = threshold
        self.categorical = categorical
        self.immutable = immutable
        self.step_direction = step_direction
        self.y_desired = y_desired
        self.set_size = set_size
        self.random_state = random_state

    def _counterfactual(self, idx, agent, cf_candidates):
        nn = NearestNeighbors(n_neighbors=1).fit(cf_candidates)
        cf = cf_candidates.iloc[nn.kneighbors(agent)[-1].squeeze()].to_frame().T
        cf.index = [idx]
        return cf

    def counterfactual(self, X, action_set=None):
        dice_ml = _optional_import("dice_ml")
        model = _ThresholdClassifier(
            classifier=self.model, threshold=self.threshold, y_desired=self.y_desired
        )
        X = X.copy()

        if not hasattr(self, "action_set_"):
            self.set_actions(X=X, action_set=action_set)

        self._backend = (
            self.model.__module__.split(".")[0]
            if self.model_backend is None
            else self.model_backend
        )

        immutable = self.immutable if self.immutable is not None else []
        categorical = self.categorical if self.immutable is not None else []

        # Set up basic elements from dice_ml
        d = dice_ml.Data(
            dataframe=pd.concat([X, pd.Series(model.predict(X), name="y")], axis=1),
            continuous_features=X.columns.drop(categorical).tolist(),
            outcome_name="y",
        )
        m = dice_ml.Model(model, backend=self.model_backend)
        exp = dice_ml.Dice(d, m)

        # Set up permitted range
        permitted_range = dict(
            self.action_set_.df[["name", "lb", "ub"]]
            .apply(lambda row: (row["name"], [row["lb"], row["ub"]]), axis=1)
            .to_list()
        )
        permitted_range = {
            k: [
                str(self.action_set_.df.set_index("name").loc[k, "variable_type"](i))
                if k in categorical
                else i
                for i in v
            ]
            for k, v in permitted_range.items()
        }

        # Do not change if the agent is over the threshold
        mask = model.predict(X) != self.y_desired
        X_ = X.iloc[mask].copy()

        # Get counterfactual candidates
        dice_exp = exp.generate_counterfactuals(
            X_,
            total_CFs=self.set_size,
            desired_class=self.y_desired,
            # features_to_vary=X.columns.drop(immutable).tolist(),
            # permitted_range=permitted_range,
            random_seed=self.random_state,
        )

        # Get counterfactuals closest to agent
        counterfactuals = pd.concat(
            [
                self._counterfactual(
                    idx, agent.to_frame().T, cf.final_cfs_df.drop(columns="y")
                )
                for (idx, agent), cf in zip(X_.iterrows(), dice_exp.cf_examples_list)
            ]
        )
        X.iloc[mask, :] = counterfactuals

        return X
