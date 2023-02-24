import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from algorec.environments import BankLoanApplication
from algorec.populations import BasePopulation
from algorec.recourse import NFeatureRecourse

RNG_SEED = 42
N_CONTINUOUS = 5
N_CAT = 2
N_AGENTS = 100

# Generate data and regression coefficients
continuous = [f"f_{i}" for i in range(N_CONTINUOUS)]
categorical = [f"cat_{i}" for i in range(N_CAT)]
rng = np.random.default_rng(RNG_SEED)
df = pd.DataFrame(
    rng.random((N_AGENTS, N_CONTINUOUS)), columns=[f"f_{i}" for i in range(N_CONTINUOUS)]
)
for i in range(N_CAT):
    df[f"cat_{i}"] = rng.integers(0, 2, 100)

y = rng.integers(0, 2, 100)
lr = LogisticRegression().fit(df, y)

# Generate elements to run simulation
population = BasePopulation(df, categorical=categorical)
recourse = NFeatureRecourse(lr, n_features=2)
environment = BankLoanApplication(
    population, recourse, threshold=.9, adaptation=.2, growth_rate=1.01, random_state=RNG_SEED
)

environment.update()

environment.run_simulation(50)
