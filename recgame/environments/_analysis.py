from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import binom, norm


class EnvironmentAnalysis:
    def __init__(self, environment):
        self.environment = environment

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

        env = self.environment

        steps = [step] if last_step is None else [s for s in range(step, last_step)]
        success_rates = []
        for step in steps:
            # Indices of agents that moved
            adapted = env._get_moving_agents(step)

            # Indices of agents above threshold (according to the last state)
            above_threshold = (
                env.metadata_[step]["score"] >= env.metadata_[step - 1]["threshold"]
            )
            above_threshold = above_threshold[above_threshold].index.to_numpy()

            # Indices of agents that moved and crossed the threshold
            candidates = np.intersect1d(adapted, above_threshold)

            # Indices of all agents with favorable outcome
            favorable_outcomes = env.metadata_[step]["outcome"].astype(bool)
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

        env = self.environment

        step = 1 if step is None else step
        last_step = max(env.metadata_.keys()) if last_step is None else last_step

        steps = [step] if last_step is None else [s for s in range(step, last_step + 1)]
        steps = [step - 1] + steps
        thresholds = [env.metadata_[step]["threshold"] for step in steps]
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
        env = self.environment

        idx = {step: meta["X"].index for step, meta in env.metadata_.items()}

        # entered_step
        entered = [
            pd.Series(i, index=idx[i][~idx[i].isin(idx[i - 1])])
            if i != 0
            else pd.Series(i, index=idx[i])
            for i in idx.keys()
        ]
        df = pd.concat(entered).to_frame("entered_step")

        # n_adaptations
        moving = [env._get_moving_agents(i) for i in idx.keys() if i != 0]
        moving = pd.Series(
            Counter([i for submoving in moving for i in submoving]),
            name="n_adaptations",
        )
        df = pd.concat([df, moving], axis=1).copy()
        df["n_adaptations"] = df["n_adaptations"].fillna(0).astype(int)

        # favorable_step
        favorable = [(i, env.metadata_[i]["outcome"]) for i in idx.keys()]
        favorable = pd.concat(
            [
                pd.Series(i, index=outcome[outcome == 1].index, name="favorable_step")
                for i, outcome in favorable
            ]
        )
        df = pd.concat([df, favorable], axis=1)

        # original_score
        df["original_score"] = df.apply(
            lambda row: env.metadata_[row["entered_step"]]["score"].loc[row.name],
            axis=1,
        )

        # final_score
        df["final_score"] = df.apply(
            lambda row: (
                env.metadata_[row["favorable_step"]]["score"].loc[row.name]
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
            adapted = env._get_moving_agents(step)
            above_threshold = (
                env.metadata_[step]["score"] >= env.metadata_[step - 1]["threshold"]
            )
            above_threshold = above_threshold[above_threshold].index.to_numpy()
            candidates = np.intersect1d(adapted, above_threshold)
            unfavorable = ~env.metadata_[step]["outcome"].astype(bool)
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
        env = self.environment

        info = {}
        for step in env.metadata_.keys():
            if step == 0:
                continue

            # Number of agents that moved
            adapted = env._get_moving_agents(step)

            # Number of agents that moved and crossed the threshold
            above_threshold = (
                env.metadata_[step]["score"] >= env.metadata_[step - 1]["threshold"]
            )
            above_threshold = above_threshold[above_threshold].index.to_numpy()
            candidates = np.intersect1d(adapted, above_threshold)

            # Number of agents with favorable outcome
            favorable_outcomes = env.metadata_[step]["outcome"].astype(bool)
            favorable_outcomes = favorable_outcomes[favorable_outcomes].index.to_numpy()

            # Indices of agents that moved and have a favorable outcome
            success = np.intersect1d(favorable_outcomes, candidates)
            success_rate = (
                success.shape[0] / candidates.shape[0]
                if candidates.shape[0] > 0
                else np.nan
            )

            # Calculate threshold drift
            threshold_prev = env.metadata_[step - 1]["threshold"]
            threshold = env.metadata_[step]["threshold"]
            threshold_drift = (threshold - threshold_prev) / threshold_prev

            # Number of new agents
            idx_prev = env.metadata_[step - 1]["X"].index
            idx = env.metadata_[step]["X"].index
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
        scores = self.environment.metadata_[0]["score"]
        return (scores >= threshold).astype(int).sum() / scores.shape[0]

    def n_new_agents_proba(self, n_above, new_agents=None, threshold=None, step=None):
        """
        Calculates the probability for at least ``n`` new agents to be above a given
        threshold within a set of newly-introduced agents, based on the distribution of
        previously added agents.
        """
        step = self.environment.step_ if step is None else step

        if threshold is None:
            threshold = self.environment.metadata_[step]["threshold"]

        if new_agents is None:
            new_agents = self.environment.metadata_[step]["growth_k"]

        p = self.new_agent_proba(threshold)
        dist = binom(new_agents, p)
        return sum([dist.pmf(i) for i in range(n_above, new_agents + 1)])

    def moving_agent_proba(self, step=None):
        """
        Calculate probability of each agent to simultaneously adapt and cross the
        threshold.
        """
        if step is None:
            step = self.environment.step_

        if type(self.environment.behavior_function) != str:
            behavior_func_name = self.environment.behavior_function.__name__.lower()
        else:
            behavior_func_name = self.environment.behavior_function.lower()

        scores = self.environment.metadata_[step]["score"]
        threshold = self.environment.metadata_[step]["threshold"]
        adaptation = self.environment.metadata_[step]["effort"]

        mask = scores < threshold
        scores = scores[mask]
        adaptation = adaptation[mask]

        if behavior_func_name.startswith("binary"):
            p = adaptation.sum() / adaptation.shape[0]
        elif behavior_func_name.startswith("continuous"):
            p = norm(loc=0, scale=adaptation).sf(threshold - scores) * 2
        else:
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
            step = self.environment.step_

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
            step = self.environment.step_

        scores = self.environment.metadata_[step]["score"]
        threshold = self.environment.metadata_[step]["threshold"]
        n_pos = self.environment.metadata_[step]["outcome"].sum()

        # Get all combinations of numbers of negative outcomes
        new_agents = list(range(self.environment.metadata_[step]["growth_k"] + 1))
        adapting_agents = list(range((scores < threshold).sum() + 1))
        combinations = np.array(
            [(i, j) for i, j in product(new_agents, adapting_agents) if i + j < n_pos]
        )

        return self._success_single_comb(self, *combinations.T, step).sum()
