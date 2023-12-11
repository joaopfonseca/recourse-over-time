import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression


def get_scaler(n_agents=10_000, n_continuous=2, bias_factor=0, mean=0, std=1/4, random_state=None):
    rng = np.random.default_rng(random_state)
    groups = pd.Series(rng.binomial(1,.5, n_agents), name="groups")
    counts = Counter(groups)
    continuous_cols = [f"f{i}" for i in range(n_continuous)]
    
    # Generate the input dataset
    X_0 = pd.DataFrame(
        rng.normal(loc=mean, scale=std, size=(counts[0], n_continuous)),
        index=groups[groups == 0].index,
        columns=continuous_cols,
    )

    X_1 = pd.DataFrame(
        rng.normal(loc=mean+bias_factor*std, scale=std, size=(counts[1], n_continuous)),
        index=groups[groups == 1].index,
        columns=continuous_cols,
    )

    X = pd.concat([X_0, X_1]).sort_index()
    return MinMaxScaler().fit(X)


def biased_data_generator(n_agents, n_continuous=2, bias_factor=0, mean=0, std=1/4, scaler=None, random_state=None):
    """
    Generate synthetic data.
    
    groups feature: 
    - 0 -> Disadvantaged group
    - 1 -> Advantaged group
    
    ``bias_factor`` varies between [0, +inf[, where 0 is completely unbiased.
    """
    rng = np.random.default_rng(random_state)
    groups = pd.Series(rng.binomial(1,.5, n_agents), name="groups")
    counts = Counter(groups)
    continuous_cols = [f"f{i}" for i in range(n_continuous)]

    # Generate the input dataset
    X_0 = pd.DataFrame(
        rng.normal(loc=mean, scale=std, size=(counts[0], n_continuous)),
        index=groups[groups == 0].index,
        columns=continuous_cols,
    )

    X_1 = pd.DataFrame(
        rng.normal(loc=mean+bias_factor*std, scale=std, size=(counts[1], n_continuous)),
        index=groups[groups == 1].index,
        columns=continuous_cols,
    )

    X = pd.concat([X_0, X_1]).sort_index()
    
    # TEST: scale continuous features
    if scaler is not None:
        X.loc[:,:] = scaler.transform(X)
    
    X = pd.concat([X, groups], axis=1)
    X = np.clip(X, 0, 1)
    
    # Generate the target
    p0 = 1 / (2 + 2*bias_factor)
    p1 = 1 - p0

    y0 = rng.binomial(1, p0, counts[0])
    y1 = rng.binomial(1, p1, counts[1])
    
    y = pd.concat(
        [
            pd.Series((y0 if val==0 else y1), index=group.index) 
            for val, group in X.groupby("groups")
        ]
    ).sort_index()

    return X, y


def fairness_metrics_viz_data(environment):
    # Get groups
    groups = pd.concat([environment.metadata_[step]["X"]["groups"] for step in environment.metadata_.keys()])
    groups = groups[~groups.index.duplicated(keep='last')].sort_index()

    # Get time for recourse
    agents_info = environment.analysis.agents_info()
    agents_info = pd.concat([agents_info, groups], axis=1)
    agents_info["time_for_recourse"] = agents_info["favorable_step"] - agents_info["entered_step"]

    results = {}
    
    # ETR - Effort to recourse
    ai_etr = agents_info.dropna(subset="favorable_step")
    ai_etr = ai_etr[ai_etr["n_adaptations"]!=0]

    ai_etr["total_effort"] = ai_etr["final_score"] - ai_etr["original_score"]
    etr = ai_etr.groupby("groups").mean()["total_effort"]
    results["etr_disparity"] = etr.loc[0] / etr.loc[1]    
    
    # TTR
    ttr = ai_etr.groupby("groups").mean()["time_for_recourse"]
    results["disparate_ttr"] = ttr.loc[0] - ttr.loc[1]

    return ai_etr


def fairness_metrics_per_time_step(environment):
    # Get groups
    groups = pd.concat([environment.metadata_[step]["X"]["groups"] for step in environment.metadata_.keys()])
    groups = groups[~groups.index.duplicated(keep='last')].sort_index()

    # Get time for recourse
    agents_info = environment.analysis.agents_info()
    agents_info = pd.concat([agents_info, groups], axis=1)
    agents_info["time_for_recourse"] = agents_info["favorable_step"] - agents_info["entered_step"]

    # Get fairness analysis
    fairness_analysis = agents_info.dropna().groupby("groups").mean()
    success_rates = environment.analysis.success_rate(filter_feature="groups")
    sns.lineplot(success_rates)
    

def fairness_metrics_overall_visualizations(environment):
    # Get groups
    groups = pd.concat([environment.metadata_[step]["X"]["groups"] for step in environment.metadata_.keys()])
    groups = groups[~groups.index.duplicated(keep='last')].sort_index()

    # Get time for recourse
    agents_info = environment.analysis.agents_info()
    agents_info = pd.concat([agents_info, groups], axis=1)
    agents_info["time_for_recourse"] = agents_info["favorable_step"] - agents_info["entered_step"]

    results = {}
    
    # ETR - Effort to recourse
    ai_etr = agents_info.dropna(subset="favorable_step")
    ai_etr = ai_etr[ai_etr["n_adaptations"]!=0]

    ai_etr["total_effort"] = ai_etr["final_score"] - ai_etr["original_score"]
    etr = ai_etr.groupby("groups").mean()["total_effort"]
    results["etr_disparity"] = etr.loc[0] / etr.loc[1]    
    
    # TTR
    ttr = ai_etr.groupby("groups").mean()["time_for_recourse"]
    results["disparate_ttr"] = ttr.loc[0] - ttr.loc[1]

    sns.boxplot(data=ai_etr, x="groups", y="total_effort")
    plt.show()
    
    sns.boxplot(data=ai_etr, x="groups", y="time_for_recourse")
    plt.show()

    return results


def fairness_metrics_overall(environment):
    # Get groups
    groups = pd.concat([environment.metadata_[step]["X"]["groups"] for step in environment.metadata_.keys()])
    groups = groups[~groups.index.duplicated(keep='last')].sort_index()

    # Get time for recourse
    agents_info = environment.analysis.agents_info()
    agents_info = pd.concat([agents_info, groups], axis=1)
    agents_info["time_for_recourse"] = agents_info["favorable_step"] - agents_info["entered_step"]

    results = {}
    
    # ETR - Effort to recourse
    ai_etr = agents_info.dropna(subset="favorable_step")
    ai_etr = ai_etr[ai_etr["n_adaptations"]!=0]

    ai_etr["total_effort"] = ai_etr["final_score"] - ai_etr["original_score"]
    etr = ai_etr.groupby("groups").mean()["total_effort"]
    results["etr_disparity"] = etr.loc[0] / etr.loc[1]    
    
    # TTR
    ttr = ai_etr.groupby("groups").mean()["time_for_recourse"]
    results["disparate_ttr"] = ttr.loc[0] - ttr.loc[1]
    
    return results

def get_scaler_hc(n_agents=10_000, bias_factor=0, mean=0, std=0.2, random_state=None, **kwargs):
    X = biased_data_generator_hc(
        n_agents, bias_factor=bias_factor, mean=mean, std=std, scaler=None, random_state=random_state, **kwargs
    )
    return MinMaxScaler().fit(X[["f0", "f1"]])


def biased_data_generator_hc(n_agents, bias_factor=0, mean=0, std=0.2, scaler=None, random_state=None, **kwargs):
    
    if "N_LOANS" in kwargs.keys():
        N_LOANS = kwargs["N_LOANS"]
        
    if "N_AGENTS" in kwargs.keys():
        N_AGENTS = kwargs["N_AGENTS"]
        
    rng = np.random.default_rng(random_state)

    # For advantaged group
    mu, sigma = mean+1.5, std
    mu2, sigma2 = mean, std
    high_perf = int((N_LOANS / (2*N_AGENTS)) * n_agents)
    X1 = rng.normal(mu, sigma, high_perf)
    X2 = rng.normal(mu2, sigma2, int((n_agents / 2) - high_perf))
    f0 = np.concatenate([X1, X2])
    f1 = rng.normal(mean, std, int(n_agents/2))
    X_adv = np.stack([f0, f1, np.ones(int(n_agents/2))], axis=1)
    
    # For disadvantaged group
    mu, sigma = mean+1.5, std
    mu2, sigma2 = mean - (bias_factor*std), std
    high_perf = int((N_LOANS / (2*N_AGENTS)) * n_agents)
    X1 = rng.normal(mu, sigma, high_perf)
    X2 = rng.normal(mu2, sigma2, int((n_agents / 2) - high_perf))
    f0 = np.concatenate([X1, X2])
    f1 = rng.normal(mean, std, int(n_agents/2))
    X_disadv = np.stack([f0, f1, np.zeros(int(n_agents/2))], axis=1)

    X = pd.DataFrame(np.concatenate([X_adv, X_disadv]), columns=["f0", "f1", "groups"])

    # TEST: scale continuous features
    if scaler is not None:
        X.loc[:,["f0", "f1"]] = scaler.transform(X[["f0", "f1"]])
    
    
    #X = np.clip(X, 0, 1)
    return X


class Ranker:
    def __init__(self, coefficients, threshold=0.5):
        self.coefficients = coefficients
        self.threshold = threshold
        
    def fit(self, X, y):
        """This is a placeholder. Should not be used."""
        return self

    def predict(self, X):
        return (self.predict_proba(X) > self.threshold).astype(int).squeeze()
    
    def predict_proba(self, X):
        self.coef_ = np.array(self.coefficients).reshape((1, -1))
        self.intercept_ = 0
        
        return np.dot(X, self.coef_.T)
    

class IgnoreGroupRanker:
    def __init__(self, coefficients, threshold=0.5, ignore_feature=None, intercept=0):
        self.coefficients = coefficients
        self.threshold = threshold
        self.ignore_feature = ignore_feature
        self.intercept = intercept
    
    def _get_X(self, X):
        return X.copy() if self.ignore_feature is None else X.drop(columns=self.ignore_feature)
    
    def fit(self, X, y):
        """This is a placeholder. Should not be used."""
        return self
    
    def predict(self, X):
        return (self.predict_proba(X) > self.threshold).astype(int).squeeze()
    
    def predict_proba(self, X):
        self.coef_ = np.array(self.coefficients).reshape((1, -1))
        self.intercept_ = self.intercept
        
        return np.dot(self._get_X(X), self.coef_.T)
    

class IgnoreGroupLR(LogisticRegression):
    def __init__(self, ignore_feature=None, **kwargs):
        super().__init__(**kwargs)
        self.ignore_feature = ignore_feature
    
    def _get_X(self, X):
        return X.copy() if self.ignore_feature is None else X.drop(columns=self.ignore_feature)
    
    def fit(self, X, y, sample_weight=None):
        """NOTE: X must be a pandas dataframe."""
        super().fit(self._get_X(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X, **kwargs):
        return super().predict(self._get_X(X), **kwargs)
    
    def predict_proba(self, X):
        return super().predict_proba(self._get_X(X))


def make_mean_sem_table(
    mean_vals,
    sem_vals=None,
    make_bold=False,
    maximum=True,
    threshold=None,
    decimals=2,
    axis=1,
):
    """
    Generate table with rounded decimals, bold maximum/minimum values or values
    above/below a given threshold, and combine mean and sem values.

    Arguments
    ---------
    mean_vals : pd.DataFrame
        Dataframe with results statistics. Must not
        contain non-indexed metadata. Supports both single and multi index.

    sem_vals : {pd.DataFrame or np.ndarray}, default=None
        Dataframe with standard errors of the means. If it is a DataFrame, must not
        contain non-indexed metadata. Supports both single and multi index.

    make_bold : bool, default=False
        If True, make bold the lowest or highest values, or values lower than, or higher
        than the passed value in ``threshold`` per row or column. If False, the
        parameters ``maximum``, ``threshold`` and ``axis`` are ignored.

    maximum : bool, default=True
        Whether to look for the highest or lowest values:

        - If True and ``threshold`` is None, boldfaces the highest value in each
          row/column.
        - If False and ``threshold`` is None, boldfaces the lowest value in each
          row/column.
        - If True and ``threshold`` is not None, boldfaces all values above the given
          threshold.
        - If False and ``threshold`` is not None, boldfaces all values below the given
          threshold.

    threshold : int or float, default=None
        Threshold to boldface values. If None, one value will be boldfaced per row or
        column. If not None, boldfaces all values above or below ``threshold``.

    decimals : int, default=2
        Number of decimal places to round each value to.

    axis : {0 or 'index', 1 or 'columns'}, default=1
        Axis along which the function is applied:

        - 0 or 'index': apply function to column.
        - 1 or 'columns': apply function to each row.

    Returns
    -------
    scores : pd.DataFrame
        Dataframe with the specified formatting.
    """

    if sem_vals is not None:
        if type(sem_vals) is np.ndarray:
            sem_vals = pd.DataFrame(
                sem_vals, index=mean_vals.index, columns=mean_vals.columns
            )

        scores = (
            mean_vals.applymap(("{:,.%sf}" % decimals).format)
            + r" $\pm$ "
            + sem_vals.applymap(("{:,.%sf}" % decimals).format)
        )
    else:
        scores = mean_vals.applymap(("{:,.%sf}" % decimals).format)

    if make_bold:
        mask = mean_vals.apply(
            lambda row: _make_bold(row, maximum, decimals, threshold, with_sem=True)[1],
            axis=axis,
        ).values

        scores.iloc[:, :] = np.where(mask, "\\textbf{" + scores + "}", scores)

    return scores
