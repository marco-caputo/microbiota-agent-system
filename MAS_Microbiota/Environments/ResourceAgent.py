from typing import Tuple
from repast4py.space import DiscretePoint as dpt
from MAS_Microbiota import Simulation
from abc import abstractmethod

from MAS_Microbiota.Environments import GridAgent


class ResourceAgent(GridAgent):
    """
    Abstract class for the agents that represent resources in the environment.
    Resources are agents that can be consumed by other agents. They can be proteins, nutrients, etc.
    All resource agents do have a context, a position in the environment and make a random movement to
    a nearby location at each step.
    Some resource agents can be removed from the environment, so they have a flag to indicate that.
    """

    def __init__(self, local_id: int, type: int, rank: int, pt: dpt, context):
        super().__init__(local_id=local_id, type=type, rank=rank, pt=pt, context=context)
        self.toRemove = False
        self.toMove = False

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
            rand_pos = Simulation.model.rng.choice(nghs_coords)
            chosen_dpt = dpt(rand_pos[0], rand_pos[1])
            Simulation.model.move(self, chosen_dpt, self.context)

    def check_if_to_move(self, permeability_check: bool = True):
        """
        Checks if the agent should move from its current environment to the brain through the bloodstream.
        This can happen whenever the agent is in the gut or microbiota and it is placed near the border of
        the environment grid.
        The movement can be conditioned by a permeability check, which is a random check that can be applied
        to the agent to determine if it can cross the barrier.

        If the permeability check is satisfied, the agent will be marked for movement through the attribute toRemove.

        :param permeability_check: A flag to indicate if the movement should be checked for permeability.
        """
        if self.pt is None:
            return
        nghs_coords = Simulation.model.ngh_finder.find(self.pt.x, self.pt.y)
        if len(nghs_coords) <= 6 and self.context in {'gut', 'microbiota'}:
            if not permeability_check:
                self.toMove = True
            elif Simulation.model.barrier_impermeability < Simulation.params["barrier_impermeability"]:
                percentage_threshold = int((Simulation.model.barrier_impermeability *
                                            Simulation.params["barrier_impermeability"]) / 100)
                choice = Simulation.model.rng.randint(0, 100)
                if choice > percentage_threshold:
                    self.toMove = True

    @abstractmethod
    def step(self):
        ...