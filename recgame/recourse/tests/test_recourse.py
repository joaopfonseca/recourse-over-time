import pytest
from itertools import product
import numpy as np
from sklearn.linear_model import LogisticRegression
from recgame.recourse import DiCE, ActionableRecourse
from recgame.utils import generate_synthetic_data
from recgame.utils._testing import all_recourse

RANDOM_SEED = 42
THRESHOLD = 0.6
N_AGENTS = 20

df, y, _ = generate_synthetic_data(
    n_agents=N_AGENTS, n_continuous=4, n_categorical=0, random_state=RANDOM_SEED
)

df_cat, y_cat, categorical = generate_synthetic_data(
    n_agents=N_AGENTS, n_continuous=4, n_categorical=2, random_state=RANDOM_SEED
)


@pytest.mark.parametrize("name, Estimator", all_recourse())
def test_all_recourse(name, Estimator):
    clf = LogisticRegression().fit(df, y)
    recourse = Estimator(model=clf, threshold=THRESHOLD)
    recourse.set_actions(df)
    recourse.action_set_.ub = 2
    recourse.action_set_.lb = -2

    counterfactuals = recourse.counterfactual(df)
    assert ~np.isnan(counterfactuals.values).any()
    assert (clf.predict_proba(counterfactuals)[:, -1] >= THRESHOLD - 1e-10).all()
    assert (counterfactuals.dtypes == df.dtypes).all()


@pytest.mark.parametrize("name, Recourse", all_recourse())
def test_categorical_features(name, Recourse):
    clf = LogisticRegression().fit(df_cat, y)

    if name == "NFeatureRecourse":
        with pytest.raises(TypeError):
            recourse = Recourse(model=clf, threshold=THRESHOLD, categorical=categorical)
    else:
        recourse = Recourse(model=clf, threshold=THRESHOLD, categorical=categorical)
        counterfactuals = recourse.counterfactual(df_cat)
        assert ~np.isnan(counterfactuals.values).any()
        assert (clf.predict_proba(counterfactuals)[:, -1] >= THRESHOLD).all()
        assert (counterfactuals.dtypes == df_cat.dtypes).all()


@pytest.mark.parametrize(
    "name, Recourse, X, immutable",
    [
        (n, r, x, i)
        for (n, r), (x, i) in product(
            all_recourse(), [(df, "f_1"), (df_cat.drop(columns="cat_1"), "cat_0")]
        )
    ],
)
def test_immutable_features(name, Recourse, X, immutable):
    clf = LogisticRegression().fit(X, y)
    clf.coef_ += .1

    if immutable == "cat_0":
        cat_coef = clf.coef_[0, X.columns == "cat_0"]
        clf.coef_[0, X.columns == "cat_0"] = 0
        clf.coef_[0, X.columns != "cat_0"] = (
            clf.coef_[0, X.columns != "cat_0"] + (cat_coef / (X.shape[1] - 1))
        )

    categorical = ["cat_0"] if "cat_0" in X.columns else None

    if name == "NFeatureRecourse" and immutable == "cat_0":
        with pytest.raises(TypeError):
            recourse = Recourse(model=clf, threshold=THRESHOLD, categorical=categorical)
        return

    recourse = Recourse(
        model=clf,
        threshold=THRESHOLD,
        categorical=categorical,
        immutable=[immutable]
    )

    counterfactuals = recourse.counterfactual(X)
    assert (counterfactuals[immutable] == X[immutable]).all()
    assert ~np.isnan(counterfactuals.values).any()
    assert (clf.predict_proba(counterfactuals)[:, -1] >= THRESHOLD - 1e-10).all()
    assert (counterfactuals.dtypes == X.dtypes).all()


@pytest.mark.parametrize("Recourse", [DiCE, ActionableRecourse])
def test_multiple_categorical_with_immutable_features(Recourse):
    immutable = "cat_1"
    clf = LogisticRegression().fit(df_cat, y)
    recourse = Recourse(
        model=clf,
        threshold=THRESHOLD,
        immutable=[immutable],
        categorical=categorical
    )
    recourse.set_actions(df_cat)
    for col in df_cat.columns.drop(categorical):
        recourse.action_set_[col].ub = 3
        recourse.action_set_[col].lb = -3

    counterfactuals = recourse.counterfactual(df_cat)
    assert (counterfactuals[immutable] == df_cat[immutable]).all()
    assert ~np.isnan(counterfactuals.values).any()
    assert (clf.predict_proba(counterfactuals)[:, -1] >= THRESHOLD).all()
    assert (counterfactuals.dtypes == df_cat.dtypes).all()
