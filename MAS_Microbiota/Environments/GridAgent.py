from abc import ABC, abstractmethod
from typing import Tuple

from repast4py.space import DiscretePoint as dpt
from repast4py.core import Agent

class GridAgent(Agent, ABC):
    """
    Abstract class for the agents that can be located in discrete positions of a
    certain grid environment.
    """

    TYPE: int

    def __init__(self, local_id: int, type: int, rank: int, pt: dpt, context: str):
        super().__init__(id=local_id, type=type, rank=rank)
        self.pt = pt
        self.context = context

    @abstractmethod
    def save(self) -> Tuple[int, dpt, str]:
        """
        Returns the agent's state in a tuple. The tuple should contain the environment-unique id
        as first element, followed by other information like the agent's state, it's position and
        its current environment name.
        :return: The agent's state in a tuple.
        """
        return self.id, self.pt, self.context

    @abstractmethod
    def step(self):
        """
        The agent's step function. This function is called at each simulation step.
        """
        pass