from abc import ABC, abstractmethod
from typing import Tuple, Any
from repast4py import context as ctx
from repast4py.core import Agent

class GridEnvironment(ABC):
    """
    Abstract class for the environment of the simulation.
    """

    NAME: str

    def __init__(self, context: ctx, grid):
        self.context = context
        self.grid = grid

    @staticmethod
    @abstractmethod
    def agent_types() -> Tuple[str, Agent, Any]:
        """
        Returns the agent types that are used in the environment.
        This method should return a tuple with the following elements:
        - The name of the parameter used to define the initial number of agents of this type in the simulation.
        - The class of the agent.
        - Additional parameters that are needed to create the agent, possibly None.

        :return: The tuple for creating the initial agents.
        """
        pass

    @abstractmethod
    def step(self):
        """
        This method should implement the logic of the environment at each step of the simulation.
        """
        pass

    def synchronize(self, restore_agent):
        """
        Synchronize the agents in the environment.
        """
        self.context.synchronize(restore_agent)

    def agents(self):
        return self.context.agents()

    def remove(self, agent):
        self.context.remove(agent)