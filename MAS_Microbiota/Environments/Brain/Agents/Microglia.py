from typing import Tuple
from repast4py import core
from repast4py.space import DiscretePoint as dpt

from MAS_Microbiota import Simulation
from MAS_Microbiota.Environments.Gut.Agents import Oligomer
class Microglia(core.Agent):
    TYPE = 6

    def __init__(self, local_id: int, rank: int, initial_state, pt: dpt, context):
        super().__init__(id=local_id, type=Microglia.TYPE, rank=rank)
        self.state = initial_state
        self.pt = pt
        self.context = context

    def save(self) -> Tuple:
        return (self.uid, self.state, self.pt.coordinates, self.context)

    # Microglia step function
    def step(self):
        nghs_coords = Simulation.model.ngh_finder.find(self.pt.x, self.pt.y)
        ngh = self.check_oligomer_nghs(nghs_coords)
        if ngh is not None:
            if self.state == Simulation.params["microglia_state"]["resting"]:
                self.state = Simulation.params["microglia_state"]["active"]
            else:
                ngh.toRemove = True

    # returns the oligomer agent in the neighborhood of the agent     
    def check_oligomer_nghs(self, nghs_coords):
        for ngh_coord in nghs_coords:
            ngh_array = Simulation.model.envs['brain']['grid'].get_agents(dpt(ngh_coord[0], ngh_coord[1]))
            for ngh in ngh_array:
                if (type(ngh) == Oligomer):
                    return ngh
        return None