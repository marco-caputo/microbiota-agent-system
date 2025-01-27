from enum import Enum, IntEnum
from typing import Tuple
from repast4py.space import DiscretePoint as dpt
from MAS_Microbiota.Environments.ResourceAgent import ResourceAgent

class SCFAType(IntEnum):
    ACETATE = 1
    BUTYRATE = 2
    PROPIONATE = 3

class SCFA(ResourceAgent):
    TYPE = 10

    def __init__(self, local_id: int, rank: int, scfa_type: SCFAType, pt: dpt, context):
        super().__init__(local_id=local_id, type=self.TYPE, rank=rank, pt=pt, context=context)
        self.scfa_type = scfa_type


    def save(self) -> Tuple:
        return (self.uid, int(self.scfa_type), self.pt.coordinates, self.context, self.toRemove)


    def step(self):
        self.random_movement()
        self.check_if_to_move(permeability_check=False)

    def BBB_impermeability_coefficient(self):
        """
        Provides a positive or negative coefficient representing the effect of the SCFA on the Blood-Brain Barrier.
        :return: 1 if the SCFA improves BBB impermeability, -1 otherwise
        """
        return -1 if self.scfa_type == SCFAType.BUTYRATE else 1

    def neuroinflammation_coefficient(self):
        """
        Provides a positive or negative coefficient representing the effect of the SCFA on neuroinflammation.
        :return: 1 if the SCFA is pro-inflammatory, -1 otherwise
        """
        return 1 if self.scfa_type == SCFAType.ACETATE else -1