from typing import Tuple
from repast4py.space import DiscretePoint as dpt
from enum import Enum
from MAS_Microbiota.Environments.ResourceAgent import ResourceAgent

class SubstrateType(Enum):
    SUGAR = 0
    CARBOYDRATE = 1
    FIBER = 2

class Substrate(ResourceAgent):
    TYPE = 9

    def __init__(self, local_id: int, type: int, sub_type: SubstrateType, rank: int, pt: dpt, context):
        super().__init__(local_id=local_id, type=type, rank=rank, pt=pt, context=context)
        self.sub_type = sub_type


    def save(self) -> Tuple:
        return (self.uid, self.sub_type, self.pt.coordinates, self.toRemove, self.context)


    def step(self):
        self.random_movement()