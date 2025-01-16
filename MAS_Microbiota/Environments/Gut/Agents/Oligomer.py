from typing import Tuple
from repast4py import core
from repast4py.space import DiscretePoint as dpt
import numpy as np

from MAS_Microbiota import Simulation

class Oligomer(core.Agent):
    TYPE = 3

    def __init__(self, local_id: int, rank: int, oligomer_name, pt: dpt, context):
        super().__init__(id=local_id, type=Oligomer.TYPE, rank=rank)
        self.name = oligomer_name
        self.pt = pt
        self.toRemove = False
        self.toMove = False
        self.context = context

    def save(self) -> Tuple:
        return (self.uid, self.name, self.pt.coordinates, self.toRemove, self.context)

    # Oligomer step function
    def step(self):
        if self.pt is None:
            return
        else:
            nghs_coords = Simulation.model.ngh_finder.find(self.pt.x, self.pt.y)
            random_index = np.random.randint(0, len(nghs_coords))
            chosen_dpt = dpt(nghs_coords[random_index][0], nghs_coords[random_index][1])
            Simulation.model.move(self, chosen_dpt, self.context)
            if len(nghs_coords) <= 6 and self.context == 'gut':
                if Simulation.model.barrier_impermeability < Simulation.params["barrier_impermeability"]:
                    percentage_threshold = int((Simulation.model.barrier_impermeability *
                                                Simulation.params["barrier_impermeability"]) / 100)
                    choice = np.random.randint(0, 100)
                    if choice > percentage_threshold:
                        self.toMove = True