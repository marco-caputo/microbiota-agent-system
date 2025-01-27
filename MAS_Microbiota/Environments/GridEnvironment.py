from abc import ABC, abstractmethod
from typing import Tuple, Any, Set
from repast4py import context as ctx
from repast4py.core import Agent

from MAS_Microbiota import Simulation


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
    def initial_agents() -> Tuple[str, Agent, Any]:
        """
        Returns the agent types that are used in the environment.
        This method should return a tuple with the following elements:
        - The name of the parameter used to define the initial number of agents of this type in the simulation.
        - The class of the agent.
        - Additional parameters that are needed to create the agent, possibly None.

        :return: The tuple for creating the initial agents.
        """
        ...

    @abstractmethod
    def step(self):
        """
        This method should implement the logic of the environment at each step of the simulation.
        """
        ...

    @abstractmethod
    def agents_to_remove(self) -> tuple[type[Agent]]:
        """
        Returns a list of agent types that should be removed from the environment if marked for removal.
        Included types of agents should implement the toRemove attribute.
        :return: A list of agent types.
        """
        ...

    def remove_agents(self, removed_ids: set[property]):
        """
        Removes the agents marked for removal from the environment. The given set of removed_ids is updated with
        the ids of the removed agents.
        :param removed_ids: A set with the ids of already removed agents.
        """
        remove_agents = [agent for agent in self.context.agents()
                         if isinstance(agent, self.agents_to_remove()) and getattr(agent, "toRemove", False)]
        for agent in remove_agents:
            if agent.uid not in removed_ids and self.context.agent(agent.uid) is not None:
                self.remove(agent)
                removed_ids.add(agent.uid)

    def make_agents_steps(self):
        """
        Let each agent in the environment perform its step.
        """
        for agent in self.context.agents():
            agent.step()

    def synchronize(self, restore_agent):
        """
        Synchronize the agents in the environment.
        """
        self.context.synchronize(restore_agent)

    def agents(self):
        return self.context.agents()

    def remove(self, agent):
        self.context.remove(agent)