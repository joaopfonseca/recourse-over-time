import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

from .swarmplot import swarm


class EnvironmentPlot:
    """
    TODO: Add documentation.
    """

    _favorable = "green"
    _unfavorable = "blue"
    _previous = "red"

    def __init__(self, environment, random_state=None):
        self.environment = environment
        self.random_state = random_state

    def fit(self, X=None):
        if X is None:
            X = self.environment.population.data

        self.is_large_dim = X.shape[-1] > 2

        if self.is_large_dim:
            self.autoencoder = MLPRegressor(
                hidden_layer_sizes=(2,),
                activation="identity",
                max_iter=1000,
                random_state=self.random_state,
            )
            self.autoencoder.fit(X, X)

            X = pd.DataFrame(
                (
                    np.dot(X.values, self.autoencoder.coefs_[0])
                    + self.autoencoder.intercepts_[0]
                ),
                index=X.index,
                columns=["Component 1", "Component 2"],
            )

        self._min_x = X.min(0)
        self._max_x = X.max(0)

        return self

    def _create_mesh_grid(self, X_curr, mesh_size):

        X_ = pd.DataFrame(
            (
                np.dot(X_curr.values, self.autoencoder.coefs_[0])
                + self.autoencoder.intercepts_[0]
            ),
            index=X_curr.index,
            columns=["Component 1", "Component 2"],
        )
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
                np.dot(mesh, self.autoencoder.coefs_[1])
                + self.autoencoder.intercepts_[1]
            )

        mesh = self.environment.model_.predict_proba(
            pd.DataFrame(mesh, columns=X_curr.columns)
        )[:, -1]
        self.mesh_ = np.reshape(mesh, mesh_size)
        return self

    def scatter(self, step=None, mesh_size=(100, 100), legend=True, title=True):
        """Visualize the population in a 2d-scatter plot."""
        if not hasattr(self, "autoencoder"):
            self.fit()

        if step is None:
            step = self.environment.step_

        df = self.environment.metadata_[step]["population"].data
        if step > 0:
            df_prev = self.environment.metadata_[step - 1]["population"].data
        threshold = self.environment.metadata_[step]["threshold"]
        mask = self.environment.model_.predict_proba(df)[:, -1] >= threshold

        self._create_mesh_grid(df, mesh_size)

        if self.is_large_dim:
            df = pd.DataFrame(
                (
                    np.dot(df.values, self.autoencoder.coefs_[0])
                    + self.autoencoder.intercepts_[0]
                ),
                index=df.index,
                columns=["Component 1", "Component 2"],
            )
            df_prev = (
                pd.DataFrame(
                    (
                        np.dot(df_prev.values, self.autoencoder.coefs_[0])
                        + self.autoencoder.intercepts_[0]
                    ),
                    index=df_prev.index,
                    columns=["Component 1", "Component 2"],
                )
                if step > 0
                else None
            )

        # Visualize probabilities
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        prob = self.mesh_

        ax.contourf(self.mesh_x, self.mesh_y, prob, levels=100, alpha=0.5, cmap="Blues")

        # Visualize agents
        if df_prev is not None:
            idx = df.index.intersection(df_prev.index)
            move = (df.loc[idx] != df_prev.loc[idx]).all(1)
            ax.plot(
                [
                    df_prev.loc[idx].loc[move].iloc[:, 0],
                    df.loc[idx].loc[move].iloc[:, 0]
                ],
                [
                    df_prev.loc[idx].loc[move].iloc[:, 1],
                    df.loc[idx].loc[move].iloc[:, 1]
                ],
                color="gray",
                alpha=0.5,
                zorder=1,
            )
            ax.scatter(
                x=df_prev.loc[idx].loc[move].iloc[:, 0],
                y=df_prev.loc[idx].loc[move].iloc[:, 1],
                alpha=0.3,
                color=self._previous,
                label="Prev. position" if legend else None,
            )

        ax.scatter(
            x=df.iloc[~mask, 0],
            y=df.iloc[~mask, 1],
            color=self._unfavorable,
            alpha=0.5,
            label="Unfavorable" if legend else None,
        )
        ax.scatter(
            x=df.iloc[mask, 0],
            y=df.iloc[mask, 1],
            color=self._favorable,
            alpha=0.5,
            label="Favorable" if legend else None,
        )
        if legend:
            ax.legend()
        if title:
            ax.set_title(f"Population at t={step}")
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        return fig, ax

    def agent_scores(self, min_step=None, max_step=None, legend=True, title=True):
        """Visualize population scores across multiple time steps."""
        if not hasattr(self, "autoencoder"):
            self.fit()

        df_list = [
            (i, metadata["population"].data, metadata["threshold"])
            for i, metadata in self.environment.metadata_.items()
        ][min_step:max_step]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        for step, df, threshold in df_list:
            threshold = self.environment.metadata_[step]["threshold"]

            prob = self.environment.model_.predict_proba(df)[:, -1]
            mask = prob >= threshold

            # Set up x coordinates to form swarm plots
            x = np.ones(df.shape[0], dtype=int) * step
            x = x + swarm(prob)

            add_labels = legend and step in [min_step, 0]

            ax.scatter(
                x=x[~mask],
                y=prob[~mask],
                color=self._unfavorable,
                alpha=0.5,
                label=("Unfavorable" if add_labels else None),
            )
            ax.scatter(
                x=x[mask],
                y=prob[mask],
                color=self._favorable,
                alpha=0.5,
                label=("Favorable" if add_labels else None),
            )

        ax.plot(
            [meta[0] for meta in df_list],
            [meta[2] for meta in df_list],
            color=self._previous,
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

        return fig, ax
