from typing import Tuple
from repast4py import core
from repast4py.space import DiscretePoint as dpt
import numpy as np

from MAS_Microbiota import Simulation
from .Microglia import Microglia

class Cytokine(core.Agent):
    TYPE = 8

    def __init__(self, local_id: int, rank: int, pt: dpt, context):
        super().__init__(id=local_id, type=Cytokine.TYPE, rank=rank)
        self.pt = pt
        self.context = context
        possible_types = [Simulation.params["cyto_state"]["pro_inflammatory"],
                          Simulation.params["cyto_state"]["non_inflammatory"]]
        random_index = np.random.randint(0, len(possible_types))
        self.state = possible_types[random_index]
        if self.state == Simulation.params["cyto_state"]["pro_inflammatory"]:
            Simulation.model.pro_cytokine += 1
        else:
            Simulation.model.anti_cytokine += 1

    def save(self) -> Tuple:
        return (self.uid, self.state, self.pt.coordinates, self.context)

    # Cytokine step function
    def step(self):
        if self.pt is None:
            return
        microglie_nghs, nghs_coords = self.get_microglie_nghs()
        if len(microglie_nghs) == 0:
            random_index = np.random.randint(0, len(nghs_coords))
            Simulation.model.move(self, dpt(nghs_coords[random_index][0], nghs_coords[random_index][1]), self.context)
        else:
            ngh_microglia = microglie_nghs[0]
            if self.state == Simulation.params["cyto_state"]["pro_inflammatory"] and ngh_microglia.state == \
                    Simulation.params["microglia_state"]["resting"]:
                ngh_microglia.state = Simulation.params["microglia_state"]["active"]
            elif self.state == Simulation.params["cyto_state"]["non_inflammatory"] and ngh_microglia.state == \
                    Simulation.params["microglia_state"]["active"]:
                ngh_microglia.state = Simulation.params["microglia_state"]["resting"]

    # returns the microglia agents in the neighborhood of the agent
    def get_microglie_nghs(self):
        nghs_coords = Simulation.model.ngh_finder.find(self.pt.x, self.pt.y)
        microglie = []
        for ngh_coords in nghs_coords:
            nghs_array = Simulation.model.envs['brain']['grid'].get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            for ngh in nghs_array:
                if (type(ngh) == Microglia):
                    microglie.append(ngh)
        return microglie, nghs_coords