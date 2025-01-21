from typing import Tuple
from repast4py.space import DiscretePoint as dpt

from MAS_Microbiota.Environments.ResourceAgent import ResourceAgent

class Protein(ResourceAgent):
    TYPE = 1

    def __init__(self, local_id: int, rank: int, protein_name, pt: dpt, context):
        super().__init__(local_id=local_id, type=Protein.TYPE, rank=rank, pt=pt, context=context)
        self.name = protein_name
        self.toCleave = False

    def save(self) -> Tuple:
        return (self.uid, self.name, self.pt.coordinates, self.toCleave, self.toRemove, self.context)

    # Protein step function
    def step(self):
        super().step()

    # changes the state of the protein agent
    def change_state(self):
        if not self.toCleave:
            self.toCleave = True