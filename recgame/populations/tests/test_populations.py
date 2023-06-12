import pytest
from recgame.populations import ActionSet
from recgame.utils import generate_synthetic_data
from recgame.utils._testing import all_population

RANDOM_SEED = 42
df, y, cat_features = generate_synthetic_data(
    n_agents=30, n_continuous=2, n_categorical=1, random_state=RANDOM_SEED
)


@pytest.mark.parametrize("name, Pop", all_population())
def test_populations(name, Pop):
    # There are categorical features that must be defined
    with pytest.raises(AssertionError):
        Pop(X=df)

    population = Pop(X=df, categorical=cat_features)
    act_set = ActionSet(X=df, default_bounds=(0, 1))
    assert act_set[cat_features[0]].variable_type == bool

    act_set[cat_features[0]].variable_type = int
    act_set["f_0"].lb = -1
    population.set_actions(action_set=act_set)
    assert population.action_set_ == act_set

    # Variable should not be of type string
    with pytest.raises(AssertionError):
        act_set[cat_features[0]].variable_type = str
