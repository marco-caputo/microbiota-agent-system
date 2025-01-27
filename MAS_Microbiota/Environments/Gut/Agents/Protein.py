from enum import IntEnum
from typing import Tuple
from repast4py.space import DiscretePoint as dpt

from MAS_Microbiota.Environments.ResourceAgent import ResourceAgent

class ProteinName(IntEnum):
    TAU = 1
    ALPHA_SYN = 2


class Protein(ResourceAgent):
    TYPE = 1

    def __init__(self, local_id: int, rank: int, protein_name: ProteinName, pt: dpt, context):
        super().__init__(local_id=local_id, type=Protein.TYPE, rank=rank, pt=pt, context=context)
        self.name = protein_name
        self.toCleave = False

    def save(self) -> Tuple:
        return (self.uid, int(self.name), self.pt.coordinates, self.toCleave, self.toRemove, self.context)

    def step(self):
        self.random_movement()

    def change_state(self):
        """
        Changes the state of the protein agent in response to an external cleaving agent.
        """
        if not self.toCleave:
            self.toCleave = True