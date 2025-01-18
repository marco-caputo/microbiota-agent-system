from typing import Tuple
from repast4py.space import DiscretePoint as dpt
import numpy as np

from MAS_Microbiota import Simulation
from MAS_Microbiota.Environments import ResourceAgent


class Oligomer(ResourceAgent):
    TYPE = 3

    def __init__(self, local_id: int, rank: int, oligomer_name, pt: dpt, context):
        super().__init__(local_id=local_id, type=Oligomer.TYPE, rank=rank, pt=pt, context=context)
        self.name = oligomer_name
        self.toMove = False

    def save(self) -> Tuple:
        return (self.uid, self.name, self.pt.coordinates, self.toRemove, self.context)

    # Oligomer step function
    def step(self):
        super().step()
        if self.pt is not None:
            nghs_coords = Simulation.model.ngh_finder.find(self.pt.x, self.pt.y)
            if len(nghs_coords) <= 6 and self.context == 'gut':
                if Simulation.model.barrier_impermeability < Simulation.params["barrier_impermeability"]:
                    percentage_threshold = int((Simulation.model.barrier_impermeability *
                                                Simulation.params["barrier_impermeability"]) / 100)
                    choice = np.random.randint(0, 100)
                    if choice > percentage_threshold:
                        self.toMove = True