from copy import deepcopy
from sklearn.base import clone
from imblearn.pipeline import make_pipeline
from .base import BaseEnvironment
from ..data_augmentation import CDA


class ModelRetrainEnvironment(BaseEnvironment):
    """
    Retrains the classifier at every step.
    """

    def _simulate(self):
        # Simulation runs as is
        super()._simulate()

        # Model is updated using the new data
        self.model_ = clone(self.model_).fit(self.X_, self.outcome_)

        # Overwrite threshold
        self.threshold_ = self.get_score_threshold()

        return self


class CDAEnvironment(BaseEnvironment):
    """
    Retrains the classifier at every step using Counterfactual Data Augmentation (CDA).
    Based on "The Robustness of Counterfactual Explanations Over Time", by Ferrario et
    al.
    """

    # def _check(self):
    #     if not hasattr(self, "model_"):
    #         self.model_ = deepcopy(self.recourse.model)

    #     super()._check()

    def _simulate(self):
        # Simulation runs as is
        super()._simulate()

        # Augment the training dataset
        cda_rec = deepcopy(self.recourse)
        cda_rec.threshold = self.threshold_
        cda_rec.model = deepcopy(self.model_)
        data_augm = CDA(recourse=cda_rec, ir=1, random_state=self.random_state)
        X_aug, y_aug = data_augm.fit_resample(self.X_, self.outcome_)

        # Model is updated using the new data
        self.model_ = clone(self.model_).fit(X_aug, y_aug)

        # Overwrite threshold
        self.threshold_ = self.get_score_threshold()

        return self
