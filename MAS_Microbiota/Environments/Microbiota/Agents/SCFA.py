from enum import Enum
from typing import Tuple
from repast4py.space import DiscretePoint as dpt
from MAS_Microbiota.Environments.ResourceAgent import ResourceAgent

class SCFAType(Enum):
    ACETATE = 0
    BUTYRATE = 1
    PROPIONATE = 2

class SCFA(ResourceAgent):
    TYPE = 10

    def __init__(self, local_id: int, type: int, scfa_type: SCFAType, rank: int, pt: dpt, context):
        super().__init__(local_id=local_id, type=type, rank=rank, pt=pt, context=context)
        self.scfa_type = scfa_type


    def save(self) -> Tuple:
        return (self.uid, self.scfa_type, self.pt.coordinates, self.toRemove, self.context)


    def step(self):
        super().step()

    def BBB_integrity_coefficient(self):
        """
        Provides a positive or negative coefficient representing the effect of the SCFA on the Blood-Brain Barrier.
        :return: 1 if the SCFA improves BBB integrity, -1 otherwise
        """
        return -1 if self.scfa_type == SCFAType.BUTYRATE else 1

    def neuroinflammation_coefficient(self):
        """
        Provides a positive or negative coefficient representing the effect of the SCFA on neuroinflammation.
        :return: 1 if the SCFA is pro-inflammatory, -1 otherwise
        """
        return 1 if self.scfa_type == SCFAType.ACETATE else -1