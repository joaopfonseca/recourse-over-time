from abc import ABC, abstractmethod
from typing import Union
from copy import deepcopy
from itertools import product
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from scipy.stats import binom, norm

from ..visualization import EnvironmentPlot


class BaseEnvironment(ABC, BaseEstimator):
    """
    Define the constraints in the environment. The central piece of the
    multi-agent analysis of algorithmic recourse.

    ``random_state`` is only relevant for visualizations or if
    ``adaptation_type in ['binary', gaussian]``. For a closed environment (no new
    agents) define ``growth_rate = 0``.

    Some relevant parameters will include:
    - [x] Population
    - [x] Decision model (not necessary)
    - [x] Threshold definition (should accept fixed or dynamic thresholds)
        * Could be defined as a function?
    - [x] Algorithmic Recourse method
    - [x] Distribution of Agents' percentage of adaptation
    - [x] Define how this percentage changes over time:
        * Increase rate: over time people get more motivated to move towards
          counterfactual
        * Decrease rate: over time people lose motivation to move towards
          counterfactual until they eventually give up
        * Mixed: Some people get more motivation, others lose it.
    - [ ] Add verbosity
    - [ ] A lot of functions are using self.model_.predict_proba. Agents' scores could
          be saved in the metadata and repurposed to save on processing time; save a
          ``self.score_`` variable, save it on metadata and use these where
          ``self.model_.predict_proba`` is being used.

    Parameters:
    - population: ``Population`` object containing the Agents' information

    - recourse: Recourse method that allows the production of counterfactuals

    - adaptation: If adaptation_type is stepwise, it represents the rate of adaptation
        of agents towards their counterfactuals. If adaptation_type is binary, the
        adaptation value determines the likelihood of adaptation.  If adaptation_type is
        binary_fixed, the adaptation value determines the number of adaptations.  If
        adaptation_type is gaussian, the adaptation value determines the flexibility of
        agents to adapt.

    - adaptation_type: can be binary, binary_fixed, stepwise, or gaussian

    - threshold: probability threshold to consider an agent's outcome as
      favorable/unfavorable

    - threshold_type: can be fixed, dynamic or absolute

    - growth_rate_type: can be absolute or relative

    - remove_winners: can be True, False

    Attributes:
    - Current step
    -
    """

    _estimator_type = "environment"

    def __init__(
        self,
        X,
        recourse,
        threshold: Union[float, int, list, np.ndarray] = 0.5,
        threshold_type: str = "fixed",
        adaptation: Union[float, int, list, np.ndarray] = 1.0,
        adaptation_type: str = "stepwise",
        growth_rate: Union[float, int, list, np.ndarray] = 1.0,
        growth_rate_type: str = "relative",
        remove_winners: bool = True,
        random_state=None,
    ):
        self.X = X
        self.recourse = recourse
        self.threshold = threshold
        self.threshold_type = threshold_type
        self.adaptation = adaptation
        self.adaptation_type = adaptation_type
        self.growth_rate = growth_rate
        self.growth_rate_type = growth_rate_type
        self.remove_winners = remove_winners
        self.random_state = random_state

        self.plot = EnvironmentPlot(self, random_state=random_state)
        self._check()
        self._save_metadata()

    def _check(self):
        # the original parameters must be kept unchanged
        if not hasattr(self, "X_"):
            self.X_ = deepcopy(self.X)

        if not hasattr(self, "step_"):
            self.step_ = 0

        if not hasattr(self.recourse, "action_set_"):
            self.recourse.set_actions(self.X_)

        if not hasattr(self, "model_"):
            self.model_ = deepcopy(self.recourse.model)

        if not hasattr(self, "growth_k_"):
            self.growth_k_ = self._update_growth_rate()

        if not hasattr(self, "threshold_"):
            self.threshold_ = self._update_threshold()

        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(self.random_state)

        if not hasattr(self, "outcome_"):
            self.outcome_, self.scores_ = self.predict(return_scores=True)

        if not hasattr(self, "adaptation_"):
            self.adaptation_ = self._update_adaptation()

        if not hasattr(self, "_max_id"):
            self._max_id = self.X_.index.max()

    def _update_threshold(self):
        threshold = (
            self.threshold[self.step_]
            if type(self.threshold) in [np.ndarray, list]
            else self.threshold
        )

        if self.threshold_type == "dynamic":
            self.threshold_index_ = int(np.round(threshold * self.X_.shape[0]))
            probabilities = self.model_.predict_proba(self.X_)[:, 1]
            threshold_ = probabilities[np.argsort(probabilities)][self.threshold_index_]

        elif self.threshold_type == "absolute":
            self.threshold_index_ = self.X_.shape[0] - threshold
            if self.threshold_index_ < 0:
                self.threshold_index_ = self.X_.shape[0] - 1

            probabilities = self.model_.predict_proba(self.X_)[:, 1]
            threshold_ = (
                probabilities[np.argsort(probabilities)][self.threshold_index_]
                if threshold != 0
                else np.nan
            )

        elif self.threshold_type == "fixed":
            threshold_ = threshold

        else:
            raise NotImplementedError()

        return threshold_

    def _update_adaptation(self):
        """
        Overwrite this method to implement custom adaptation functions.

        Must return a pd.Series object (where the index corresponds to the population's
        indices).

        NOTE: for the "binary_fixed" case to work properly, this parameter must ALWAYS
        be the last one to be updated.
        """
        adaptation = (
            self.adaptation[self.step_]
            if type(self.adaptation) == [np.ndarray, list]
            else self.adaptation
        )

        if self.adaptation_type == "gaussian":
            # adaptation are the standard deviations for each agent's gaussian
            scores = self.model_.predict_proba(self.X_)[:, 1]
            threshold = self.threshold_
            x = threshold - scores
            adaptation = (adaptation) / (10 * np.exp(5 * x))

        elif self.adaptation_type == "binary":
            adaptation = self._rng.binomial(1, adaptation, self.X_.shape[0])

        elif self.adaptation_type == "binary_fixed":
            idx = np.where(self.scores_ < self.threshold_)[0]

            # If this is false, we assume all remaining agents adapt
            if idx.shape[0] > int(adaptation):
                idx = self._rng.choice(idx, size=int(adaptation), replace=False)

            adaptation = np.zeros(self.X_.shape[0], dtype=int)
            adaptation[idx] = 1

        elif self.adaptation_type == "stepwise":
            # No action needed - elif included for clarification
            pass

        return pd.Series(adaptation, index=self.X_.index)

    def _update_growth_rate(self):
        """Must return the number of agents to add."""
        growth_rate = (
            self.growth_rate[self.step_]
            if type(self.growth_rate) == [np.ndarray, list]
            else self.growth_rate
        )

        if self.growth_rate_type == "absolute":
            growth_k = int(np.round(growth_rate))
        elif self.growth_rate_type == "relative":
            growth_k = int(np.round(self.X_.shape[0] * growth_rate))
        return growth_k

    def predict(self, X=None, step=None, return_scores=False):
        """
        Produces the outcomes for a single agent or a population.

        If ``return_scores=True``, the function will return both the outcome and scores.

        TODO: Refactor if statement
        """

        if step is None:
            threshold = self.threshold_
            if self.threshold_type in ["dynamic", "absolute"]:
                threshold_idx = self.threshold_index_
            if X is None:
                X = self.X_
        else:
            threshold = self.metadata_[step]["threshold"]
            if self.threshold_type in ["dynamic", "absolute"]:
                threshold_idx = self.metadata_[step]["threshold_index"]
            if X is None:
                X = self.metadata_[step]["X"]

        # Output should consider threshold and return a single value per
        # observation
        probs = self.model_.predict_proba(X)[:, 1]
        if self.threshold_type in ["dynamic", "absolute"]:
            pred = np.zeros(probs.shape, dtype=int)
            idx = np.argsort(probs)[threshold_idx:]
            pred[idx] = 1
        else:
            pred = (probs >= threshold).astype(int)

        if return_scores:
            return (
                pd.Series(pred, index=X.index),
                pd.Series(probs, index=X.index),
            )
        else:
            return pd.Series(pred, index=X.index)

    def counterfactual(self, X=None, step=None):
        """
        Get the counterfactual examples.
        """
        if X is None:
            X = self.X_ if step is None else self.metadata_[step]["X"]

        # Ensure threshold is up-to-date
        threshold = (
            self.threshold_ if step is None else self.metadata_[step]["threshold"]
        )
        if np.isnan(threshold):
            return X
        else:
            self.recourse.threshold = threshold
            return self.recourse.counterfactual(X)

    def _counterfactual_gaussian_vectors(self, factuals, counterfactuals):
        score_change = np.abs(self._rng.normal(loc=0, scale=self.adaptation_))
        curr_scores = self.model_.predict_proba(self.X_)[:, 1]
        target_scores = np.clip(curr_scores + score_change, 0, 0.999)

        # Get base vectors
        base_vectors = counterfactuals - factuals
        base_dist = np.linalg.norm(base_vectors, axis=1).reshape(-1, 1)
        base_vectors = base_vectors / base_dist

        # Get updated intersect
        intercept = self.model_.intercept_ - np.log(target_scores / (1 - target_scores))

        # Get multiplier
        multiplier = (
            -intercept - np.dot(factuals, self.model_.coef_.T).squeeze()
        ) / np.dot(base_vectors, self.model_.coef_.T).squeeze()

        cf_vectors = multiplier.reshape(-1, 1) * base_vectors
        cf_vectors.loc[base_dist == 0] = 0
        return cf_vectors

    def _counterfactual_stepwise_vectors(self, factuals, counterfactuals):
        """Used only for stepwise and gaussian adaptation approaches."""

        # Compute avg distance to counterfactual
        base_dist = np.linalg.norm(counterfactuals - factuals, axis=1).reshape(-1, 1)
        avg_dist = base_dist[~np.isnan(base_dist)].mean()

        # Get base counterfactual vectors
        cf_vectors = avg_dist * ((counterfactuals - factuals) / base_dist)

        # Compute adaptation ratio
        # Compute euclidean distance for each agent
        return cf_vectors

    def _save_metadata(self):
        """
        Store metadata for each step. Currently incomplete.
        """

        if not hasattr(self, "metadata_"):
            self.metadata_ = {}

        self.metadata_[self.step_] = {
            "X": deepcopy(self.X_),
            "adaptation": deepcopy(self.adaptation_),
            "outcome": self.outcome_,
            "score": self.scores_,
            "threshold": self.threshold_,
            "growth_k": self.growth_k_,
        }
        if self.threshold_type in ["dynamic", "absolute"]:
            self.metadata_[self.step_]["threshold_index"] = self.threshold_index_

    @abstractmethod
    def add_agents(self, n_agents):
        # Generate data
        # Add info to population
        pass

    def remove_agents(self, mask):
        indices = np.where(~mask.astype(bool))[0]
        self.X_ = self.X_.iloc[indices]
        self.adaptation_ = self.adaptation_.iloc[indices]
        return self

    def adapt_agents(self, X):
        """
        Applies changes to the population according to the adaptation type chosen.

        Returns the data for the adapted individuals, the original data, and the
        counterfactuals.
        """
        # Get factuals + counterfactuals
        factuals = self.X_
        counterfactuals = self.counterfactual()

        # Get adaptation rate for all agents
        if self.adaptation_type in ["binary", "binary_fixed"]:
            cf_vectors = counterfactuals - factuals

        elif self.adaptation_type == "stepwise":
            cf_vectors = self._counterfactual_stepwise_vectors(
                factuals, counterfactuals
            )

        elif self.adaptation_type == "gaussian":
            cf_vectors = self._counterfactual_gaussian_vectors(
                factuals, counterfactuals
            )
            new_factuals = factuals + cf_vectors
            new_factuals = np.clip(
                new_factuals,
                self.recourse.action_set_.lb,
                self.recourse.action_set_.ub,
            )
            return new_factuals, factuals, counterfactuals

        else:
            raise NotImplementedError()

        # Update existing agents' feature values
        new_factuals = factuals + self.adaptation_.values.reshape(-1, 1) * cf_vectors
        new_factuals = np.clip(
            new_factuals,
            self.recourse.action_set_.lb,
            self.recourse.action_set_.ub,
        )
        return new_factuals, factuals, counterfactuals

    def update(self):
        """
        Moves environment to the next time step.

        Updates threshold, agents in population, model (?)
        The update must:
        - Remove the Agents with favorable outcomes
        - Add new Agents
        - Remove Agents with adverse outcomes that would give up
        """

        # Remove agents with favorable outcome from previous step
        if self.remove_winners:
            self.remove_agents(self.outcome_)

        # Adapt agents
        new_factuals, factuals, counterfactuals = self.adapt_agents(X=self.X_)
        self.X_ = new_factuals

        # Add new agents
        self._new_agents = self.add_agents(self.growth_k_)
        self._new_agents.index = range(
            self._max_id + 1, self._max_id + self._new_agents.shape[0] + 1
        )
        self.X_ = pd.concat([self.X_, self._new_agents])
        self._max_id = self.X_.index.max()

        # Update variables
        self.growth_k_ = self._update_growth_rate()
        self.threshold_ = self._update_threshold()
        self.outcome_, self.scores_ = self.predict(return_scores=True)
        self.adaptation_ = self._update_adaptation()

        # Update metadata and step number
        self.step_ += 1
        self._save_metadata()

        return self

    def run_simulation(self, n_steps):
        """
        Simulates ``n`` runs through the environment.
        """
        for _ in range(n_steps):
            self.update()
        return self

    def _get_moving_agents(self, step):
        """Get indices of agents that adapted between ``step-1`` and ``step``."""
        if step == 0:
            raise IndexError("Agents cannot move at the initial state (``step=0``).")

        adapted = (
            (self.metadata_[step - 1]["adaptation"] > 0)
            & (~self.metadata_[step - 1]["outcome"].astype(bool))
            & (
                self.model_.predict_proba(self.metadata_[step - 1]["X"])[:, 1]
                < self.metadata_[step - 1]["threshold"]
            )
        )
        return adapted[adapted].index.values

    # NOTE: THIS SECTION ONWARDS IS DEDICATED TO ANALYSIS.
    #       THESE METHODS SHOULD BE IN THEIR OWN SUBMODULE.
    def success_rate(self, step, last_step=None):
        """
        For an agent to move, they need to have adaptation > 0, unfavorable outcome and
        score < threshold.

        If nan, no agents adapted (thus no rate is calculated).

        NOTE: DEPRECATED; REMOVE SOON.
        """
        if step == 0:
            raise IndexError(
                "Cannot calculate success rate at ``step=0`` (agents have not adapted "
                "yet)."
            )

        steps = [step] if last_step is None else [s for s in range(step, last_step)]
        success_rates = []
        for step in steps:
            # Indices of agents that moved
            adapted = self._get_moving_agents(step)

            # Indices of agents above threshold (according to the last state)
            above_threshold = (
                self.metadata_[step]["score"] >= self.metadata_[step - 1]["threshold"]
            )
            above_threshold = above_threshold[above_threshold].index.to_numpy()

            # Indices of agents that moved and crossed the threshold
            candidates = np.intersect1d(adapted, above_threshold)

            # Indices of all agents with favorable outcome
            favorable_outcomes = self.metadata_[step]["outcome"].astype(bool)
            favorable_outcomes = favorable_outcomes[favorable_outcomes].index.to_numpy()

            # Indices of agents that moved and have a favorable outcome
            success = np.intersect1d(favorable_outcomes, candidates)
            success_rate = (
                success.shape[0] / candidates.shape[0]
                if candidates.shape[0] > 0
                else np.nan
            )

            success_rates.append(success_rate)
        return np.array(success_rates)

    def threshold_drift(self, step=None, last_step=None):
        """
        Calculate threshold variations across time steps.

        NOTE: DEPRECATED; REMOVE SOON.
        """

        if step == 0:
            raise IndexError("Cannot get threshold drift at ``step=0``.")

        step = 1 if step is None else step
        last_step = max(self.metadata_.keys()) if last_step is None else last_step

        steps = [step] if last_step is None else [s for s in range(step, last_step + 1)]
        steps = [step - 1] + steps
        thresholds = [self.metadata_[step]["threshold"] for step in steps]
        threshold_drift = [
            (thresholds[i] - thresholds[i - 1]) / thresholds[i - 1]
            for i in range(1, len(thresholds))
        ]
        return np.array(threshold_drift)

    def agents_info(self):
        """
        Return a dataframe with number of adaptations performed by each agent that
        entered the environment.

        Contains the following features:
        - n_adaptations: number of times the agent adapted.
        - entered_step: step in which the agent entered the environment.
        - favorable_step: step in which the agent obtained a positive outcome. If nan,
          the agent never received a positive outcome.
        - original_score: score of the agent upon entering the environment.
        - final_score: score of the agent when obtaining the positive outcome.
        - n_failures: number of times agent adapted, crossed the threshold but didn't
          obtain a positive outcome.
        """
        idx = {step: meta["X"].index for step, meta in self.metadata_.items()}

        # entered_step
        entered = [
            pd.Series(i, index=idx[i][~idx[i].isin(idx[i - 1])])
            if i != 0
            else pd.Series(i, index=idx[i])
            for i in idx.keys()
        ]
        df = pd.concat(entered).to_frame("entered_step")

        # n_adaptations
        moving = [self._get_moving_agents(i) for i in idx.keys() if i != 0]
        moving = pd.Series(
            Counter([i for submoving in moving for i in submoving]),
            name="n_adaptations",
        )
        df = pd.concat([df, moving], axis=1).copy()
        df["n_adaptations"] = df["n_adaptations"].fillna(0).astype(int)

        # favorable_step
        favorable = [(i, self.metadata_[i]["outcome"]) for i in idx.keys()]
        favorable = pd.concat(
            [
                pd.Series(i, index=outcome[outcome == 1].index, name="favorable_step")
                for i, outcome in favorable
            ]
        )
        df = pd.concat([df, favorable], axis=1)

        # original_score
        df["original_score"] = df.apply(
            lambda row: self.metadata_[row["entered_step"]]["score"].loc[row.name],
            axis=1,
        )

        # final_score
        df["final_score"] = df.apply(
            lambda row: (
                self.metadata_[row["favorable_step"]]["score"].loc[row.name]
                if not np.isnan(row["favorable_step"])
                else np.nan
            ),
            axis=1,
        )

        # n_failures
        df["n_failures"] = 0
        for step in idx.keys():
            if step == 0:
                continue
            adapted = self._get_moving_agents(step)
            above_threshold = (
                self.metadata_[step]["score"] >= self.metadata_[step - 1]["threshold"]
            )
            above_threshold = above_threshold[above_threshold].index.to_numpy()
            candidates = np.intersect1d(adapted, above_threshold)
            unfavorable = ~self.metadata_[step]["outcome"].astype(bool)
            unfavorable = unfavorable[unfavorable].index.to_numpy()
            failed = np.intersect1d(unfavorable, candidates)
            df.loc[failed, "n_failures"] += 1
        return df

    def steps_info(self):
        """
        Return a dataframe with information regarding each time step in the environment.

        Contains the following features:
        - n_adapted: number of agents that adapted.
        - n_candidates: number of agents that adapted and crossed the previous time
          step's threshold.
        - favorable_outcomes: number of favorable outcomes.
        - success_rate: percentage of favorable outcomes within agents that adapted and
          crossed the threshold.
        - threshold: threshold value.
        - threshold_drift: percentage change between thresholds.
        - new_agents: number of new agents
        - new_agents_proba: probability of a single new agent to be above the threshold.
        - moving_agent_proba: mean likelihood of an agent to adapt and cross the
          threshold.
        - success_proba: probability of an agent adapting towards its counterfactual to
          achieve a positive outcome.
        """
        info = {}
        for step in self.metadata_.keys():
            if step == 0:
                continue

            # Number of agents that moved
            adapted = self._get_moving_agents(step)

            # Number of agents that moved and crossed the threshold
            above_threshold = (
                self.metadata_[step]["score"] >= self.metadata_[step - 1]["threshold"]
            )
            above_threshold = above_threshold[above_threshold].index.to_numpy()
            candidates = np.intersect1d(adapted, above_threshold)

            # Number of agents with favorable outcome
            favorable_outcomes = self.metadata_[step]["outcome"].astype(bool)
            favorable_outcomes = favorable_outcomes[favorable_outcomes].index.to_numpy()

            # Indices of agents that moved and have a favorable outcome
            success = np.intersect1d(favorable_outcomes, candidates)
            success_rate = (
                success.shape[0] / candidates.shape[0]
                if candidates.shape[0] > 0
                else np.nan
            )

            # Calculate threshold drift
            threshold_prev = self.metadata_[step - 1]["threshold"]
            threshold = self.metadata_[step]["threshold"]
            threshold_drift = (threshold - threshold_prev) / threshold_prev

            # Number of new agents
            idx_prev = self.metadata_[step - 1]["X"].index
            idx = self.metadata_[step]["X"].index
            new_agents = idx[~idx.isin(idx_prev)].shape[0]

            # Probability of achieving a favorable outcome

            # Create entry for dataframe
            info[step] = {
                "n_adapted": adapted.shape[0],
                "n_candidates": candidates.shape[0],
                "favorable_outcomes": favorable_outcomes.shape[0],
                "success_rate": success_rate,
                "threshold": threshold,
                "threshold_drift": threshold_drift,
                "new_agents": new_agents,
                "new_agents_proba": self.new_agent_proba(threshold),
                "moving_agent_proba": self.moving_agent_proba(step=step).mean(),
                "success_proba": self.success_proba(step=step),
            }
        return pd.DataFrame(info).T

    def new_agent_proba(self, threshold):
        """
        Calculates the probability for a new agent to be above a given threshold,
        based on the distribution of previously added agents.
        """
        scores = self.metadata_[0]["score"]
        return (scores >= threshold).astype(int).sum() / scores.shape[0]

    def n_new_agents_proba(self, n_above, new_agents=None, threshold=None, step=None):
        """
        Calculates the probability for at least ``n`` new agents to be above a given
        threshold within a set of newly-introduced agents, based on the distribution of
        previously added agents.
        """
        step = self.step_ if step is None else step

        if threshold is None:
            threshold = self.metadata_[step]["threshold"]

        if new_agents is None:
            new_agents = self.metadata_[step]["growth_k"]

        p = self.new_agent_proba(threshold)
        dist = binom(new_agents, p)
        return sum([dist.pmf(i) for i in range(n_above, new_agents + 1)])

    def moving_agent_proba(self, step=None):
        """
        Calculate probability of each agent to simultaneously adapt and cross the
        threshold.
        """
        if step is None:
            step = self.step_

        scores = self.metadata_[step]["score"]
        threshold = self.metadata_[step]["threshold"]
        adaptation = self.metadata_[step]["adaptation"]

        mask = scores < threshold
        scores = scores[mask]
        adaptation = adaptation[mask]

        if self.adaptation_type in ["binary", "binary_fixed"]:
            p = adaptation.sum() / adaptation.shape[0]
        elif self.adaptation_type == "gaussian":
            p = norm(loc=0, scale=adaptation).sf(threshold - scores) * 2
        elif self.adaptation_type == "stepwise":
            raise NotImplementedError()

        return pd.Series(p, index=scores.index)

    def n_moving_agents_proba(self, n_above, step=None):
        """
        Calculates the probability for at least ``n`` agents to move above a given
        threshold within the set of agents in the environment.

        NOTE: Continuity correction is applied, i.e., the output corresponds to
        ``P(X >= n_above-0.5)``.
        """

        if step is None:
            step = self.step_

        p = self.moving_agent_proba(step)
        mean = p.sum()
        std = np.sqrt(np.sum(p * (1 - p)))

        # Applying continuity correction
        return norm(loc=mean, scale=std).sf(n_above - 0.5)

    @np.vectorize
    def _success_single_comb(self, new, ada, step):
        p_new = self.n_new_agents_proba(new, step) - self.n_new_agents_proba(
            new + 1, step
        )
        p_ada = self.n_moving_agents_proba(ada, step) - self.n_moving_agents_proba(
            ada + 1, step
        )
        return p_new * p_ada

    def success_proba(self, step=None):
        """
        Calculate the probability of an agent adapting exactly towards its counterfactual
        to receive a positive outcome, i.e., calculates
        ``P(pos_outcome | delta_score = threshold - score)``.
        """
        if step is None:
            step = self.step_

        scores = self.metadata_[step]["score"]
        threshold = self.metadata_[step]["threshold"]
        n_pos = self.metadata_[step]["outcome"].sum()

        # Get all combinations of numbers of negative outcomes
        new_agents = list(range(self.metadata_[step]["growth_k"] + 1))
        adapting_agents = list(range((scores < threshold).sum() + 1))
        combinations = np.array(
            [(i, j) for i, j in product(new_agents, adapting_agents) if i + j < n_pos]
        )

        return self._success_single_comb(self, *combinations.T, step).sum()
