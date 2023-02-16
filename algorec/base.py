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
    - [] Population
    - [] Decision model
    - [] Threshold definition (which should accept fixed or dynamic thresholds)
        * Could be defined as a function?
    - [] Agorithmic Recourse method

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

    List of features to implement:
    - [x] Agents' personal info (data)
    - [x] categorical features
    - [x] immutable features
    - [x] step direction
    - [] Internalize the definition of the action set
    - [] Distribution of Agents' percentage of adaptation
    - [] Define how this percentage changes over time:
        * Increase rate: over time people get more motivated to move towards
          counterfactual
        * Decrease rate: over time people lose motivation to move towards
          counterfactual until they eventually give up
        * Mixed: Some people get more motivation, others lose it.
    - [] Amount of new Agents per update
    - [] Amount of Agents with an adverse outcome that give up (leave the
      Population)
    - [] Generator (for new agents)
    - [] Population objects should be iterable. Selecting one element should
         return an Agent.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        y_desired: Union[int, str] = 1,
        categorical: Union[list, np.ndarray, dict] = None,
        immutable: Union[list, np.ndarray, dict] = None,
        step_direction: Union[list, np.ndarray, dict] = None,
    ):
        self.data = data
        self.y_desired = y_desired
        self.categorical = categorical
        self.immutable = immutable
        self.step_direction = step_direction

        # self.action_set = self._action_set

    def action_set(self, action_set=None):
        """
        To be configured with the ActionSet object from the
        ``actionable-recourse`` library.

        NOTE: ``recourse``'s ActionSet forces upper and lower bounds to be
        between 0 and 1. This limitation should be removed in the future.
        """
        if action_set is not None:
            self._action_set = action_set
        else:
            pass
        return self

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
