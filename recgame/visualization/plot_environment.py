import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from matplotlib.colors import ListedColormap
from mlresearch.utils import feature_to_color

from .swarmplot import swarm


class EnvironmentPlot:
    """
    TODO: Add documentation.
    """

    _favorable = ListedColormap(["green"])
    _unfavorable = ListedColormap(["blue"])
    _previous = ListedColormap(["red"])

    def __init__(self, environment, random_state=None):
        self.environment = environment
        self.random_state = random_state

    def fit(self, X=None):
        if X is None:
            X = self.environment.X

        self.is_large_dim = X.shape[-1] > 2

        if self.is_large_dim:
            self._autoencoder = MLPRegressor(
                hidden_layer_sizes=(2,),
                activation="identity",
                max_iter=1000,
                random_state=self.random_state,
            )
            self._autoencoder.fit(X, X)

            X = pd.DataFrame(
                (
                    np.dot(X.values, self._autoencoder.coefs_[0])
                    + self._autoencoder.intercepts_[0]
                ),
                index=X.index,
                columns=["Component 1", "Component 2"],
            )

        self._min_x = X.min(0)
        self._max_x = X.max(0)

        return self

    def _create_mesh_grid(self, X_curr, mesh_size, step):
        if self.is_large_dim:
            X_ = pd.DataFrame(
                (
                    np.dot(X_curr.values, self._autoencoder.coefs_[0])
                    + self._autoencoder.intercepts_[0]
                ),
                index=X_curr.index,
                columns=["Component 1", "Component 2"],
            )
        else:
            X_ = X_curr.copy()
        _min_x_curr = X_.min(0)
        _max_x_curr = X_.max(0)

        mins = np.array([_min_x_curr, self._min_x]).min(0)
        maxs = np.array([_max_x_curr, self._max_x]).max(0)

        self.mesh_x, self.mesh_y = [
            np.array(
                np.linspace(
                    mins[i] - 0.1 * (maxs[i] - mins[i]),
                    maxs[i] + 0.1 * (maxs[i] - mins[i]),
                    mesh_size[i],
                )
            )
            for i in range(2)
        ]
        mesh = np.stack(np.meshgrid(self.mesh_x, self.mesh_y), axis=2)
        mesh = np.reshape(mesh, (mesh_size[0] * mesh_size[1], 2))

        if self.is_large_dim:
            mesh = (
                np.dot(mesh, self._autoencoder.coefs_[1])
                + self._autoencoder.intercepts_[1]
            )

        mesh = self.environment.metadata_[step]["model"].predict_proba(
            pd.DataFrame(mesh, columns=X_curr.columns)
        )[:, -1]
        self.mesh_ = np.reshape(mesh, mesh_size)
        return self

    def _get_cmaps(
        self, df, color_feature, cmap_favorable, cmap_unfavorable, cmap_previous
    ):
        colfeat = (
            df[color_feature].values
            if color_feature is not None
            else np.ones(df.shape[0])
        )
        cmap_favorable = self._favorable if cmap_favorable is None else cmap_favorable
        cmap_unfavorable = (
            self._unfavorable if cmap_unfavorable is None else cmap_unfavorable
        )
        cmap_previous = self._previous if cmap_previous is None else cmap_previous

        favorable = pd.Series(
            feature_to_color(colfeat, cmap=self._favorable), index=df.index
        )
        unfavorable = pd.Series(
            feature_to_color(colfeat, cmap=self._unfavorable), index=df.index
        )
        previous = pd.Series(
            feature_to_color(colfeat, cmap=self._previous), index=df.index
        )
        return favorable, unfavorable, previous

    def scatter(
        self,
        step=None,
        mesh_size=(100, 100),
        legend=True,
        title=True,
        ax=None,
        color_feature=None,
        cmap_favorable=None,
        cmap_unfavorable=None,
        cmap_previous=None,
    ):
        """Visualize the population in a 2d-scatter plot."""
        if not hasattr(self, "_autoencoder"):
            self.fit()

        if step is None:
            step = self.environment.step_

        df = self.environment.metadata_[step]["X"]
        if step > 0:
            df_prev = self.environment.metadata_[step - 1]["X"]

        outcome = self.environment.metadata_[step]["outcome"].astype(bool)

        self._create_mesh_grid(df, mesh_size, step)

        # Get colors according to color_feature
        favorable, unfavorable, previous = self._get_cmaps(
            df, color_feature, cmap_favorable, cmap_unfavorable, cmap_previous
        )

        # Project data into 2 dimensions
        if self.is_large_dim:
            df = pd.DataFrame(
                (
                    np.dot(df.values, self._autoencoder.coefs_[0])
                    + self._autoencoder.intercepts_[0]
                ),
                index=df.index,
                columns=["Component 1", "Component 2"],
            )
            df_prev = (
                pd.DataFrame(
                    (
                        np.dot(df_prev.values, self._autoencoder.coefs_[0])
                        + self._autoencoder.intercepts_[0]
                    ),
                    index=df_prev.index,
                    columns=["Component 1", "Component 2"],
                )
                if step > 0
                else None
            )

        # Visualize probabilities
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 10))

        prob = self.mesh_

        ax.contourf(self.mesh_x, self.mesh_y, prob, levels=100, alpha=0.5, cmap="Blues")

        # Visualize agents
        if step > 0:
            move = self.environment._get_moving_agents(step)

            # Plot movement lines
            ax.plot(
                [df_prev.loc[move].iloc[:, 0], df.loc[move].iloc[:, 0]],
                [df_prev.loc[move].iloc[:, 1], df.loc[move].iloc[:, 1]],
                color="gray",
                alpha=0.5,
                zorder=1,
            )

            # Plot previous location for moving agents
            ax.scatter(
                x=df_prev.loc[move].iloc[:, 0],
                y=df_prev.loc[move].iloc[:, 1],
                alpha=0.3,
                color=previous.loc[move],
                label="Prev. position" if legend else None,
            )

        # Plot all current agents with unfavorable outcome
        ax.scatter(
            x=df.iloc[~outcome.values, 0],
            y=df.iloc[~outcome.values, 1],
            color=unfavorable.iloc[~outcome.values],
            alpha=0.5,
            label="Unfavorable" if legend else None,
        )

        # Plot all current agents with favorable outcome
        ax.scatter(
            x=df.iloc[outcome.values, 0],
            y=df.iloc[outcome.values, 1],
            color=favorable.iloc[outcome.values],
            alpha=0.5,
            label="Favorable" if legend else None,
        )
        if legend:
            ax.legend()
        if title:
            ax.set_title(f"Population at t={step}")
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        return ax

    def agent_scores(
        self,
        min_step=None,
        max_step=None,
        legend=True,
        title=True,
        ax=None,
        ref_model_step=None,
        color_feature=None,
        cmap_favorable=None,
        cmap_unfavorable=None,
        cmap_previous=None,
    ):
        """Visualize population scores across multiple time steps."""
        if not hasattr(self, "_autoencoder"):
            self.fit()

        metadata_ = self.environment.metadata_
        min_step = min_step if min_step is not None else 0
        max_step = max_step if max_step is not None else len(metadata_.keys())
        steps_list = list(range(min_step, max_step))

        if ref_model_step is None:
            df_list = [
                (
                    i,
                    metadata_[i]["X"],
                    metadata_[i]["threshold"],
                    metadata_[i]["outcome"].astype(bool),
                    metadata_[i]["score"],
                )
                for i in steps_list
            ]
        else:
            df_list = [
                (
                    i,
                    # X
                    metadata_[i]["X"],
                    # Threshold
                    metadata_[ref_model_step]["model"]
                    .predict_proba(metadata_[i]["X"])[:, 1][
                        metadata_[i]["outcome"].astype(bool)
                    ]
                    .min(),
                    # Outcome
                    metadata_[i]["outcome"].astype(bool),
                    # Score
                    metadata_[ref_model_step]["model"].predict_proba(metadata_[i]["X"])[
                        :, 1
                    ],
                )
                for i in steps_list
            ]

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 10))

        for step, df, threshold, outcome, prob in df_list:
            # Get colors according to color_feature
            favorable, unfavorable, previous = self._get_cmaps(
                df, color_feature, cmap_favorable, cmap_unfavorable, cmap_previous
            )

            # Set up x coordinates to form swarm plots
            x = np.ones(df.shape[0], dtype=int) * step
            x = x + swarm(prob)

            add_labels = legend and step in [min_step, 0]

            ax.scatter(
                x=x[~outcome.values],
                y=prob[~outcome.values],
                color=unfavorable[~outcome.values],
                alpha=0.5,
                label=("Unfavorable" if add_labels else None),
            )
            ax.scatter(
                x=x[outcome.values],
                y=prob[outcome.values],
                color=favorable[outcome.values],
                alpha=0.5,
                label=("Favorable" if add_labels else None),
            )

        ax.plot(
            [meta[0] for meta in df_list],
            [meta[2] for meta in df_list],
            color="red",
            label="Threshold",
            alpha=0.3,
        )

        if title:
            ax.set_title(f"Population scores over {len(df_list)} time steps")
        if legend:
            ax.legend()
        ax.set_xticks([meta[0] for meta in df_list])
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")

        return ax

    def scores_histogram(self, step=None, color_feature=None, **kwargs):
        """
        Plots the histogram of the scores at ``step``.

        See documentation for plt.hist in order to pass kwargs
        """
        if step is None:
            step = self.environment.step_

        model = self.environment.metadata_[step]["model"]

        return plt.hist(
            model.predict_proba(self.environment.metadata_[step]["X"])[:, -1],
            **kwargs,
        )

    def scores_kde(self, step=None, **kwargs):
        """
        Plots the KDE of the scores at ``step`` using Gaussian kernels.

        See documentation for pandas.DataFrame.plot.kde in order to pass kwargs
        """
        if step is None:
            step = self.environment.step_

        model = self.environment.metadata_[step]["model"]

        return pd.Series(
            model.predict_proba(self.environment.metadata_[step]["X"])[:, -1]
        ).plot.kde(**kwargs)

    def population_size(
        self,
        min_step=None,
        max_step=None,
        legend=True,
        title=True,
        stacked=True,
        ax=None,
        **kwargs,
    ):
        """
        plots the number of agents per time step

        See documentation for pandas.DataFrame.plot.bar in order to pass kwargs
        """

        pop_list = [
            (i, metadata["X"].shape[0], metadata["outcome"].sum())
            for i, metadata in self.environment.metadata_.items()
        ][min_step:max_step]

        df = pd.DataFrame(pop_list, columns=["step", "pop", "favorable"])
        df.set_index("step", inplace=True)
        df["unfavorable"] = df["pop"] - df["favorable"]
        df.drop(columns="pop", inplace=True)
        ax = df[["unfavorable", "favorable"]].plot.bar(stacked=stacked, ax=ax, **kwargs)

        if title:
            ax.set_title(f"Population size over {len(pop_list)} time steps")
        if legend:
            ax.legend()
        ax.set_xlabel("Step")
        ax.set_ylabel("Number of agents")

        return ax
