import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from recourse import ActionSet

from algorec import BaseEnvironment, BasePopulation, ActionableRecourse

rng = np.random.default_rng(42)
df = pd.DataFrame(rng.random((100, 3)), columns=["a", "b", "c"])
y = rng.integers(0, 2, 100)

lr = LogisticRegression().fit(df.values, y)
y_pred = lr.predict(df)

# Testing a single counterfactual
population = BasePopulation(data=df, y_desired=1)

# Explicitly defining an action set should not be necessary in the future
act_set = ActionSet(X=df, y_desired=1)
population.action_set(act_set)

recourse_algo = ActionableRecourse(lr, threshold=0.65, flipset_size=100)
cfs = recourse_algo.counterfactuals(population)


# Test a population within an environment - not working yet
population = BasePopulation(data=rng.random((100, 3)))
env = BaseEnvironment(
    population=population,
    decision_model=lr,
    threshold_definition=20,  # A fixed number of observations
    recourse_method=ActionableRecourse(),
)
