from typing import Union
import numpy as np
import pandas as pd
from .base import BaseEnvironment


class FairEnvironment(BaseEnvironment):
    def __init__(
        self,
        X,
        recourse,
        data_source_func,
        group_feature,
        nbins=10,
        adv_group_val=1,
        threshold: Union[float, int, list, np.ndarray] = 0.5,
        threshold_type: str = "absolute",  # "fixed", "relative", "absolute"
        adaptation: Union[float, int, list, np.ndarray] = 1.0,
        behavior_function: str = "continuous_flexible",
        # binary_constant, binary_flexible, continuous_constant, continuous_flexible
        growth_rate: Union[float, int, list, np.ndarray] = 1.0,
        growth_rate_type: str = "relative",  # "absolute", "relative"
        remove_winners: bool = True,
        random_state: int = None,
    ):
        self.group_feature = group_feature
        self.nbins = nbins
        self.adv_group_val = adv_group_val
        super().__init__(
            X=X,
            recourse=recourse,
            data_source_func=data_source_func,
            threshold=threshold,
            threshold_type=threshold_type,
            adaptation=adaptation,
            behavior_function=behavior_function,
            growth_rate=growth_rate,
            growth_rate_type=growth_rate_type,
            remove_winners=remove_winners,
            random_state=random_state,
        )

    def get_score_threshold(self, step=None):
        """
        Gets the score threshold ($s_t$).

        NOTE: Formerly _update_threshold
        """
        if step is not None:
            return self.metadata_[step]["threshold"]

        threshold = (
            self.threshold[self.step_]
            if type(self.threshold) in [np.ndarray, list]
            else self.threshold
        )

        if self.threshold_type == "relative":
            self.threshold_index_ = int(np.round(threshold * self.X_.shape[0]))
        elif self.threshold_type == "absolute":
            self.threshold_index_ = self.X_.shape[0] - threshold
        elif self.threshold_type == "fixed":
            probabilities = self.model_.predict_proba(self.X_)[:, -1]
            self.threshold_index_ = (probabilities <= threshold).sum()
            return threshold
        else:
            raise NotImplementedError()

        if self.threshold_index_ < 0:
            self.threshold_index_ = self.X_.shape[0] - 1

        # probabilities = self.model_.predict_proba(self.X_)[:, 1]
        outcome, probabilities = self.outcome(self.X_, return_scores=True)
        X = self.X_.copy()
        X["scores"] = probabilities
        threshold_ = (
            X[outcome.astype(bool)]
            .groupby(self.group_feature)["scores"]
            .nsmallest(1)
            .max()
        )

        return threshold_

    def outcome(self, X=None, step=None, return_scores=False):
        """
        Produces the outcomes for a single agent or a population.

        If ``return_scores=True``, the function will return both the outcome and scores.

        NOTE: Formerly predict
        """
        # if step is None:
        #     step = self.step_

        threshold_idx = (
            self.metadata_[step]["threshold_index"]
            if step is not None
            else self.threshold_index_
        )

        model = self.metadata_[step]["model"] if step is not None else self.model_

        if X is None:
            X = self.metadata_[step]["X"] if step is not None else self.X_
            # effort = self.metadata_[step]["effort"] if step is not None else self.effort_

        # Output should consider threshold and return a single value per
        # observation
        probs = model.predict_proba(X)[:, -1]

        # Fairness adjustment
        # effort = (
        #     self.metadata_[self.step_]["score"] - self.metadata_[self.step_ - 1]["score"]
        # )
        # df_fair = pd.concat(
        #     [
        #         effort.to_frame("effort"),
        #         pd.Series(probs, index=X.index, name="probs"),
        #         X[self.group_feature]
        #     ],
        #     axis=1
        # ).dropna()
        # df_fair["effort_bins"] = pd.cut(df_fair["effort"], self.nbins)

        # Fairness adjustment V2
        X = X.copy()
        n_loans = X.shape[0] - threshold_idx
        t = int(np.floor(n_loans / 2))

        X["score"] = probs
        idx = (
            X[[self.group_feature, "score"]]
            .groupby(["groups"])["score"]
            .nlargest(t)
            .index.get_level_values(1)
            .to_numpy()
        )

        pred = pd.Series(np.zeros(probs.shape, dtype=int), index=X.index)
        pred.loc[idx] = 1

        if return_scores:
            return (
                pred,
                pd.Series(probs, index=X.index),
            )
        else:
            return pred
