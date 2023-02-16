import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from algorec import BasePopulation, ActionableRecourse

rng = np.random.default_rng(42)
df = pd.DataFrame(rng.random((100, 4)), columns=["a", "b", "c", "d"])
y = rng.integers(0, 2, 100)

lr = LogisticRegression().fit(df, y)
y_pred = lr.predict(df)

# Testing a single counterfactual
population = BasePopulation(
    data=df,
    y_desired=1,
    categorical=None,
    immutable=["b"],
    step_direction={"a": -1}
)

# TODO: test for y_desired = 0

recourse_algo = ActionableRecourse(lr, threshold=0.65, flipset_size=100)
cfs = recourse_algo.counterfactuals(population)
