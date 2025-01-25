from enum import Enum
from typing import Tuple

from repast4py.space import DiscretePoint as dpt

from MAS_Microbiota import Simulation
from MAS_Microbiota.Environments import ResourceAgent


class NeurotransmitterType(Enum):
    DOPAMINE = 1
    SEROTONIN = 2
    NOREPINEPHRINE = 3

class Neurotransmitter(ResourceAgent):

    TYPE = 13

    def __init__(self, local_id: int, rank: int, neurotrans_type : NeurotransmitterType, pt: dpt, context):
        super().__init__(local_id=local_id, type=Neurotransmitter.TYPE, rank=rank, pt=pt, context=context)
        self.neurotrans_type = neurotrans_type
        self.age = 0

    def save(self) -> Tuple:
        return (self.uid, self.neurotrans_type, self.pt.coordinates, self.context)

    # Neurotransmitter step function
    def step(self):
        self.random_movement()
        self.age += 1
        if self.age > Simulation.params["neurotrans_max_age"]:
            self.toRemove = True

