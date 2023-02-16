from abc import ABC, abstractmethod
from typing import Union
from recourse import ActionSet
import numpy as np
import pandas as pd


class BaseEnvironment(ABC):
    """
    Define the constraints in the environment. The central piece of the
    multi-agent analysis of algorithmic recourse.

    Some relevant parameters will include:
    - [x] Population
    - [] Decision model
    - [] Threshold definition (which should accept fixed or dynamic thresholds)
        * Could be defined as a function?
    - [x] Agorithmic Recourse method
    - [] Distribution of Agents' percentage of adaptation
    - [] Define how this percentage changes over time:
        * Increase rate: over time people get more motivated to move towards
          counterfactual
        * Decrease rate: over time people lose motivation to move towards
          counterfactual until they eventually give up
        * Mixed: Some people get more motivation, others lose it.

    Attributes:
    - Current step
    -
    """

    def __init__(self, population, model, recourse, threshold):
        self.population = population
        self.model = model
        self.recourse = recourse
        self.threshold = threshold

    def decision(self, agent):
        """Produces the outcomes for a single agent or a population."""
        pass

    def counterfactual(self, agent):
        """Get the counterfactual examples."""
        pass

    def update(self):
        """Moves environment to the next timestep"""
        if not hasattr(self, "step_"):
            self.step_ = 1
        pass


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
                X=self.data,
                y_desired=self.y_desired,
                default_bounds=(0, 1)
            )
            for col in self.data.columns:
                action_set[col].ub = self.data[col].max()
                action_set[col].lb = self.data[col].min()
                if col in immutable:
                    action_set[col].actionable = False
                if col in step.keys():
                    action_set[col].step_direction = self.step_direction[col]
                if col in categorical:
                    action_set[col].variable_type = int

        self._action_set = action_set

        return self

    def set_params(self, **kwargs):
        """
        This should be used to add/update parameters. Use same approach as
        sklearn's.
        """
        pass

    # @abstractmethod
    # def set_generator(self):
    #     """Generator (for new agents)"""
    #     pass

    # @abstractmethod
    # def update(self):
    #     """
    #     Moves population to the next timestep.

    #     The update must:
    #     - Remove the Agents with favorable outcomes
    #     - Add new Agents
    #     - Remove Agents with adverse outcomes that would give up
    #     """
    #     pass
