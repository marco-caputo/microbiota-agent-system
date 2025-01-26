from enum import IntEnum
from typing import Tuple
from repast4py.space import DiscretePoint as dpt

from MAS_Microbiota import Simulation
from MAS_Microbiota.Environments import GridAgent

class ExternalInputType(IntEnum):
    DIET = 1
    ANTIBIOTICS = 2
    STRESS = 3

class ExternalInput(GridAgent):

    TYPE = 4

    def __init__(self, local_id: int, rank: int, input_type: ExternalInputType, pt: dpt, context):
        super().__init__(local_id=local_id, type=ExternalInput.TYPE, rank=rank, pt=pt, context=context)
        self.input_type = input_type

    def save(self) -> Tuple:
        return (self.uid, int(self.input_type), self.pt.coordinates, self.context)

    # External input step function
    def step(self):
        if Simulation.model.epithelial_barrier_impermeability >= Simulation.params["epithelial_barrier"]["permeability_threshold_stop"]:
            def adjust_bacteria(good_bacteria_factor, pathogenic_bacteria_factor):
                to_remove = int((Simulation.model.microbiota_good_bacteria_class *
                                 Simulation.model.rng.uniform(0, good_bacteria_factor)) / 100)
                Simulation.model.microbiota_good_bacteria_class -= to_remove #TODO: ammazzare batteri qui
                to_add = int((Simulation.params["microbiota_pathogenic_bacteria_class"] *
                              Simulation.model.rng.uniform(0, pathogenic_bacteria_factor)) / 100)
                Simulation.model.microbiota_pathogenic_bacteria_class += to_add

            if self.input_type == ExternalInputType.DIET:
                adjust_bacteria(3, 3)
            elif self.input_type == ExternalInputType.ANTIBIOTICS:
                adjust_bacteria(5, 2)
            else:
                adjust_bacteria(3, 3)