from typing import Tuple
from repast4py.space import DiscretePoint as dpt
from enum import IntEnum

from MAS_Microbiota import Simulation
from MAS_Microbiota.Environments.ResourceAgent import ResourceAgent

class SubstrateType(IntEnum):
    SUGAR = 1
    CARBOHYDRATE = 2
    FIBER = 3

class Substrate(ResourceAgent):
    TYPE = 9

    def __init__(self, local_id: int, rank: int, sub_type: SubstrateType, pt: dpt, context):
        super().__init__(local_id=local_id, type=self.TYPE, rank=rank, pt=pt, context=context)
        self.sub_type = sub_type
        self.age = 0


    def save(self) -> Tuple:
        return (self.uid, int(self.sub_type), self.pt.coordinates, self.context, self.toRemove)


    def step(self):
        self.random_movement()
        self.age += 1
        if self.age > Simulation.params["substrate_max_age"]:
            self.toRemove = True