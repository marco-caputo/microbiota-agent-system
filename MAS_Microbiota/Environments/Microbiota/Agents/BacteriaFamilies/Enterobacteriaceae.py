from typing import List

from MAS_Microbiota.Environments.Brain.Agents.Precursor import PrecursorType
from MAS_Microbiota.Environments.Microbiota.Agents import Bacterium, SCFAType
from repast4py.space import DiscretePoint as dpt

from MAS_Microbiota.Environments.Microbiota.Agents.Substrate import SubstrateType


class Enterobacteriaceae(Bacterium):

    def __init__(self, local_id: int, rank: int, pt: dpt, context: str):
        super().__init__(local_id=local_id, rank=rank, pt=pt, context=context)

    def fermentable_substrates(self) -> List[SubstrateType]:
        return [SubstrateType.SUGAR]

    def fermentable_precursors(self) -> List[PrecursorType]:
        return []

    def produced_scfa(self) -> List[SCFAType]:
        return [SCFAType.ACETATE]

    def consumable_scfa(self) -> List[SCFAType]:
        return self.produced_scfa()

    def produced_precursors(self) -> List[PrecursorType]:
        return [PrecursorType.TRYPTOPHAN, PrecursorType.TYROSINE]

    def can_release_bacteriocins(self) -> bool:
        return False

    def can_move(self) -> bool:
        return True

    def causes_inflammation(self) -> bool:
        return True