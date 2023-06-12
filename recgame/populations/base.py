from abc import ABC
from sklearn.base import BaseEstimator
import pandas as pd


class BasePopulation(ABC, BaseEstimator):
    """
    Defines the population of agents within the environment.

    A population consists of a set of Agents.

    Actionable recourse supports only binary categorical features.

    This is intended to be a more "static" object, which can be manipulated
    via an Environment object.

    List of features to implement:
    - [x] Agents' personal info (data)
    - [x] categorical features
    - [x] immutable features
    - [x] step direction
    - [x] Definition the action set internally
    - [x] Amount of new Agents per update
    - [] Amount of Agents with an adverse outcome that give up (leave the
      Population)
    - [] Generator (for new agents)
    - [] Population objects should be iterable. Selecting one element should
         return an Agent.
    - [] Allow entering data as numpy array (?)
    """

    _estimator_type = "population"

    def __init__(
        self,
        X: pd.DataFrame,
    ):
        self.X = X
