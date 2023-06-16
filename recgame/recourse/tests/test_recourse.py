import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from recgame.utils import generate_synthetic_data
from recgame.utils._testing import all_recourse

RANDOM_SEED = 42
THRESHOLD = 0.52

df, y, _ = generate_synthetic_data(
    n_agents=50, n_continuous=4, n_categorical=0, random_state=RANDOM_SEED
)


@pytest.mark.parametrize("name, Estimator", all_recourse())
def test_all_recourse(name, Estimator):
    clf = LogisticRegression().fit(df, y)
    recourse = Estimator(model=clf, threshold=THRESHOLD)
    counterfactuals = recourse.counterfactual(df)
    assert ~np.isnan(counterfactuals.values).any()
    assert (clf.predict_proba(counterfactuals)[:, -1] >= THRESHOLD).all()
