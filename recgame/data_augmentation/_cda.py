"""
Counterfactual Data Augmentation.
"""
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
from imblearn.over_sampling.base import BaseOverSampler
from ..recourse import DiCE


class CDA(BaseOverSampler):
    """
    Performs Counterfactual Data Augmentation using a given recourse method to ensure the
    training data matches the imbalance ratio passed in ``ir``.

    ir : imbalance ratio of the resulting dataset. ir = (fav. class + CD) / (Unf. class)
    """

    def __init__(self, recourse=None, ir=1, random_state=None):
        super(CDA, self).__init__(sampling_strategy="auto")
        self.recourse = recourse
        self.ir = ir
        self.random_state = random_state

    def _validate_random_state(self):
        self.random_state_ = check_random_state(self.random_state)
        return self

    def _validate_recourse(self, X, y):
        if self.recourse is not None:
            self.recourse_ = self.recourse
        else:
            model = LogisticRegression(random_state=self.random_state).fit(X, y)
            self.recourse_ = DiCE(model=model, random_state=self.random_state)

        return self

    def _get_n_samples(self, y):
        """
        Get the number of counterfactuals to add to the training data.
        """
        counts = Counter(y)
        pos = counts[self.recourse.y_desired]
        neg = y.shape[0] - pos
        return int(self.ir * neg - pos)

    def _fit_resample(self, X, y, sample_weight=None):
        # TODO: Check whether to keep sample_weight

        self._validate_random_state()._validate_recourse(X, y)

        categorical = (
            self.recourse_.categorical if self.recourse_.categorical is not None else []
        )
        if hasattr(self.recourse_.model, "feature_names_in_"):
            column_names = self.recourse_.model.feature_names_in_
        else:
            column_names = list(range(X.shape[1]))

        X_neg = X[y != self.recourse_.y_desired]
        X_neg = pd.DataFrame(X_neg.astype(float), columns=column_names)
        X_neg[categorical] = X_neg[categorical].astype(int).astype(str)

        X_cf = self.recourse_.counterfactual(X_neg).values

        # Sample from X_cf to ensure the IR requirement is met
        n_samples = self._get_n_samples(y)
        if n_samples < 0:
            X_cf = np.empty((0, X_cf.shape[1]))
        elif X_cf.shape[0] > n_samples:
            X_cf = X_cf[
                self.random_state_.choice(X_cf.shape[0], size=n_samples, replace=False)
            ]

        y_cf = np.ones(X_cf.shape[0]) * self.recourse_.y_desired

        X_resampled = np.concatenate([X, X_cf.astype(float)])
        y_resampled = np.concatenate([y, y_cf])

        return X_resampled, y_resampled
