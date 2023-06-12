import numpy as np
import pandas as pd
from typing import Union
from .action_set import ActionSet
from .base import BasePopulation


class Population(BasePopulation):
    """
    Contains an Action Set.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y_desired: Union[int, str] = 1,
        categorical: Union[list, np.ndarray] = None,
        immutable: Union[list, np.ndarray] = None,
        step_direction: dict = None,
    ):
        super().__init__(X=X)

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
                X=self.X, y_desired=self.y_desired, default_bounds=(0, 1)
            )
            for col in self.X.columns:
                if col in immutable:
                    action_set[col].actionable = False

                if col in step.keys():
                    action_set[col].step_direction = self.step_direction[col]

                if col in categorical:
                    action_set[col].variable_type = int
                else:
                    action_set[col].ub = self.X[col].max()
                    action_set[col].lb = self.X[col].min()

        self.action_set_ = action_set

        return self
