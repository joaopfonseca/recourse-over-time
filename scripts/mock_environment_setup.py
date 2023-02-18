import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from algorec import (
    BaseEnvironment,
    BasePopulation,
    ActionableRecourse,
)

rng = np.random.default_rng(42)
df = pd.DataFrame(rng.random((100, 4)), columns=["a", "b", "c", "d"])
df["cat_1"] = rng.integers(0, 2, 100)
y = rng.integers(0, 2, 100)

lr = LogisticRegression().fit(df, y)

# Test an environment
population = BasePopulation(
    data=df,
    # step_direction={"a": 1, "b": -1},
    # immutable=["c"],
    categorical=["cat_1"],
)

recourse = ActionableRecourse(
    model=lr,
    threshold=.6
)

environment = BaseEnvironment(
    population=population,
    recourse=recourse,
    threshold=.6,
)

assert environment.step_ == 0
