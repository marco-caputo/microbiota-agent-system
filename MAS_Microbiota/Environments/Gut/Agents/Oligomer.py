from typing import Tuple
from repast4py.space import DiscretePoint as dpt

from MAS_Microbiota.Environments import ResourceAgent
from MAS_Microbiota.Environments.Gut.Agents import ProteinName


class Oligomer(ResourceAgent):
    TYPE = 3

    def __init__(self, local_id: int, rank: int, oligomer_name: ProteinName, pt: dpt, context):
        super().__init__(local_id=local_id, type=Oligomer.TYPE, rank=rank, pt=pt, context=context)
        self.name = oligomer_name
        self.toMove = False

    def save(self) -> Tuple:
        return (self.uid, int(self.name), self.pt.coordinates, self.toRemove, self.context)

    def step(self):
        self.random_movement()
        self.check_if_to_move(permeability_check=True)