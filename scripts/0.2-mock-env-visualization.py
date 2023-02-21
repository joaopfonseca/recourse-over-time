import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

from algorec import BasePopulation, BaseEnvironment
from algorec.recourse import ActionableRecourse

rng = np.random.default_rng(42)
df = pd.DataFrame(rng.random((100, 2)), columns=["a", "b"])
y = rng.integers(0, 2, 100)

lr = LogisticRegression().fit(df, y)


def visualize_decision_boundary(df, model, mesh_size=(100, 100), threshold=0.5):
    x, y = [
        np.array(np.linspace(df.min(0)[i] - 1, df.max(0)[i] + 1, mesh_size[i]))
        for i in range(2)
    ]
    mesh = np.stack(np.meshgrid(x, y), axis=2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Visualize model
    prob = lr.predict_proba(
        np.reshape(mesh, (mesh_size[0] * mesh_size[1], 2))
    )[:, -1] < threshold
    prob = np.reshape(prob, mesh_size)
    ax.contourf(x, y, prob, alpha=0.5, cmap="Blues")
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], ax=ax)
    ax.yaxis.set_ticklabels([])
    return fig, ax


population = BasePopulation(data=df)
recourse = ActionableRecourse(model=lr, threshold=0.65)
environment = BaseEnvironment(population=population, recourse=recourse, threshold=0.65)

for i in range(4):
    visualize_decision_boundary(
        environment.population_.data,
        lr,
        (1000, 1000),
        threshold=0.65
    )
    environment.update()
