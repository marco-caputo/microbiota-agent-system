from typing import Tuple
from repast4py import core
from repast4py.space import DiscretePoint as dpt
import numpy as np
from MAS_Microbiota import Simulation
from abc import ABC, abstractmethod


class ResourceAgent(ABC, core.Agent):
    """
    Abstract class for the agents that represent resources in the environment.
    Resources are agents that can be consumed by other agents. They can be proteins, nutrients, etc.
    All resource agents do have a context, a position in the environment and make a random movement to
    a nearby location at each step.
    """

    def __init__(self, local_id: int, type: int, rank: int, pt: dpt, context):
        super().__init__(id=local_id, type=type, rank=rank)
        self.pt = pt
        self.context = context
        self.toRemove = False

    @abstractmethod
    def save(self) -> Tuple:
        pass

    def random_movement(self):
        """
        Moves the agent to a random neighbour cell.
        """
        if self.pt is None:
            return
        else:
            nghs_coords = Simulation.model.ngh_finder.find(self.pt.x, self.pt.y)
            random_index = np.random.randint(0, len(nghs_coords))
            chosen_dpt = dpt(nghs_coords[random_index][0], nghs_coords[random_index][1])
            Simulation.model.move(self, chosen_dpt, self.context)

    # Protein step function
    @abstractmethod
    def step(self):
        self.random_movement()