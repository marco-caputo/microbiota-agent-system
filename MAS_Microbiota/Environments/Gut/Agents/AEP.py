from enum import IntEnum
from typing import Tuple
from repast4py.space import DiscretePoint as dpt
import numpy as np
from .Protein import Protein

from MAS_Microbiota import Simulation
from ... import GridAgent

class AEPState(IntEnum):
    ACTIVE = 1
    HYPERACTIVE = 2

class AEP(GridAgent):
    TYPE = 0

    def __init__(self, local_id: int, rank: int, pt: dpt, context):
        super().__init__(local_id=local_id, type=AEP.TYPE, rank=rank, pt=pt, context=context)
        self.state = AEPState.ACTIVE

    def save(self) -> Tuple:
        return (self.uid, int(self.state), self.pt.coordinates, self.context)

    # returns True if the agent is hyperactive, False otherwise
    def is_hyperactive(self):
        return self.state == AEPState.HYPERACTIVE

    # AEP step function
    def step(self):
        if self.pt is None:
            return
        nghs_coords = Simulation.model.ngh_finder.find(self.pt.x, self.pt.y)
        protein = self.percepts(nghs_coords)
        if protein is not None:
            if (self.is_hyperactive() == True):
                self.cleave(protein)
        else:
            random_index = np.random.randint(0, len(nghs_coords))
            Simulation.model.move(self, dpt(nghs_coords[random_index][0], nghs_coords[random_index][1]), self.context)

    # returns the protein agent in the neighborhood of the agent
    def percepts(self, nghs_coords):
        for ngh_coords in nghs_coords:
            nghs_array = Simulation.model.envs['gut'].grid.get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            for ngh in nghs_array:
                if type(ngh) == Protein:
                    return ngh
        return None

        # cleaves the protein agent

    def cleave(self, protein):
        protein.change_state()