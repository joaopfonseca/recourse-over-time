from abc import ABC, abstractmethod
from .action_set import ActionSet
from typing import Union
import numpy as np
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
    - [] Amount of new Agents per update
    - [] Amount of Agents with an adverse outcome that give up (leave the
      Population)
    - [] Generator (for new agents)
    - [] Population objects should be iterable. Selecting one element should
         return an Agent.
    - [] Allow entering data as numpy array (?)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y_desired: Union[int, str] = 1,
        categorical: Union[list, np.ndarray] = None,
        immutable: Union[list, np.ndarray] = None,
        step_direction: dict = None,
    ):
        self.data = data
        self.y_desired = y_desired
        self.categorical = categorical
        self.immutable = immutable
        self.step_direction = step_direction

        self.set_actions()

    def set_actions(self, action_set=None):
        """
        To be configured with the ActionSet object from the
        ``actionable-recourse`` library.
        """

        categorical = [] if self.categorical is None else self.categorical
        immutable = [] if self.immutable is None else self.immutable
        step = {} if self.step_direction is None else self.step_direction

        if action_set is None:
            action_set = ActionSet(
                X=self.data, y_desired=self.y_desired, default_bounds=(0, 1)
            )
            for col in self.data.columns:
                if col in immutable:
                    action_set[col].actionable = False

                if col in step.keys():
                    action_set[col].step_direction = self.step_direction[col]

                if col in categorical:
                    action_set[col].variable_type = int
                else:
                    action_set[col].ub = self.data[col].max()
                    action_set[col].lb = self.data[col].min()

        self.action_set_ = action_set

        return self

    def set_params(self, **kwargs):
        """
        This should be used to add/update parameters. Use same approach as
        sklearn's.
        """
        pass
