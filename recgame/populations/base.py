from abc import ABC
import pandas as pd

# class Agent(ABC):
#     """
#     Defines a single agent playing within the environment.
#
#     Some relevant parameters will include:
#     - Agent's personal info
#     - Percentage of adaptation towards counterfactual
#
#     Attributes:
#     - status: exited, out of or included in population
#     - outcome: None, favorable, adverse
#     """
#
#     def __init__(self):
#         pass
#
#     def update(self):
#         """Moves agent to the next timestep"""
#         pass


class BasePopulation(ABC):
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

    def set_params(self, **kwargs):
        """
        This should be used to add/update parameters. Use same approach as
        sklearn's.
        """
        pass

    def get_params(self):
        pass
