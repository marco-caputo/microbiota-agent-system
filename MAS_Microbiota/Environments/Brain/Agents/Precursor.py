from enum import Enum, IntEnum

from MAS_Microbiota.Environments import ResourceAgent
from .Neurotransmitter import NeurotransmitterType

class PrecursorType(IntEnum):
    TRYPTOPHAN = 1
    TYROSINE = 2

    def associated_neurotransmitters(self):
        if self == PrecursorType.TRYPTOPHAN:
            return [NeurotransmitterType.SEROTONIN]
        else:
            return [NeurotransmitterType.DOPAMINE, NeurotransmitterType.NOREPINEPHRINE]


class Precursor(ResourceAgent):

    TYPE = 14

    TRYPTOPHAN_TYPE = 1
    TYROSINE_TYPE = 2

    def __init__(self, local_id: int, rank: int, precursor_type: PrecursorType, pt, context):
        super().__init__(local_id=local_id, type=Precursor.TYPE, rank=rank, pt=pt, context=context)
        self.precursor_type = precursor_type

    def save(self):
        return (self.uid, self.precursor_type, self.pt.coordinates, self.context)

    def step(self):
        super().step()