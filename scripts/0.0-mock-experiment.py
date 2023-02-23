import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from algorec.populations import BasePopulation
from algorec.recourse import ActionableRecourse, NFeatureRecourse

rng = np.random.default_rng(42)
df = pd.DataFrame(rng.random((100, 4)), columns=["a", "b", "c", "d"])
y = rng.integers(0, 2, 100)

lr = LogisticRegression().fit(df, y)
y_pred = lr.predict_proba(df)[:, -1] > 0.6

# Testing counterfactuals
population = BasePopulation(
    data=df, y_desired=1, categorical=None, immutable=["b"], step_direction={"a": -1}
)

# TODO: test for y_desired = 0

recourse = ActionableRecourse(lr, threshold=0.6, flipset_size=100)
cfs = recourse.counterfactual(population)

recourse = NFeatureRecourse(lr, threshold=0.6, n_features=None)
cfs = recourse.counterfactual(population)
