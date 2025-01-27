from enum import IntEnum
from typing import Tuple
from repast4py.space import DiscretePoint as dpt
import numpy as np

from MAS_Microbiota import Simulation
from .Microglia import MicrogliaState, Microglia
from ... import GridAgent


class CytokineState(IntEnum):
    PRO_INFLAMMATORY = 1
    NON_INFLAMMATORY = 2


class Cytokine(GridAgent):
    TYPE = 8

    def __init__(self, local_id: int, rank: int, pt: dpt, context):
        super().__init__(local_id=local_id, type=Cytokine.TYPE, rank=rank, pt=pt, context=context)
        self.state = Simulation.model.rng.choice(list(CytokineState))
        if self.state == CytokineState.PRO_INFLAMMATORY:
            Simulation.model.pro_cytokine += 1
        else:
            Simulation.model.anti_cytokine += 1

    def save(self) -> Tuple:
        return (self.uid, int(self.state), self.pt.coordinates, self.context)

    def step(self):
        if self.pt is None:
            return
        microglie_nghs, nghs_coords = self.get_microglie_nghs()
        if len(microglie_nghs) == 0:
            random_index = np.random.randint(0, len(nghs_coords))
            Simulation.model.move(self, dpt(nghs_coords[random_index][0], nghs_coords[random_index][1]), self.context)
        else:
            ngh_microglia = microglie_nghs[0]
            if self.state == CytokineState.PRO_INFLAMMATORY and ngh_microglia.state == MicrogliaState.RESTING:
                ngh_microglia.state = MicrogliaState.ACTIVE
            elif self.state == CytokineState.NON_INFLAMMATORY and ngh_microglia.state == MicrogliaState.ACTIVE:
                ngh_microglia.state = MicrogliaState.RESTING

    # returns the microglia agents in the neighborhood of the agent
    def get_microglie_nghs(self):
        nghs_coords = Simulation.model.ngh_finder.find(self.pt.x, self.pt.y)
        microglie = []
        for ngh_coords in nghs_coords:
            nghs_array = Simulation.model.envs['brain'].grid.get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            for ngh in nghs_array:
                if (type(ngh) == Microglia):
                    microglie.append(ngh)
        return microglie, nghs_coords