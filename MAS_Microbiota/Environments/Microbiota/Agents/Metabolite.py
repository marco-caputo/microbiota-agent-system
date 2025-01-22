from typing import Tuple

from MAS_Microbiota.Environments import ResourceAgent


class Metabolite(ResourceAgent):
    def save(self) -> Tuple:
        pass

    def step(self):
        pass

    def __init__(self, uid, context, local_id: int, type: int, rank: int, pt: dpt):
        super().__init__(local_id, type, rank, pt, context)
        self.uid = uid
        self.context = context
        self.toRemove = False