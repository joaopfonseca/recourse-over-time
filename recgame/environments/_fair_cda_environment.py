from copy import deepcopy
from sklearn.base import clone
from ._fair_environment import FairEnvironment
from ..data_augmentation import CDA


class FairCDAEnvironment(FairEnvironment):
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
