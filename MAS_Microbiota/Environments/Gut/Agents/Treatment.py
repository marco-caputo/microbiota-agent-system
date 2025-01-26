from enum import Enum
from typing import Tuple
from repast4py.space import DiscretePoint as dpt
import numpy as np

from MAS_Microbiota import Simulation
from MAS_Microbiota.Environments import GridAgent

class TreatmentType(Enum):
    DIET = 1
    PROBIOTICS = 2

class Treatment(GridAgent):

    TYPE = 5

    def __init__(self, local_id: int, rank: int, treatment_type: TreatmentType, pt: dpt, context):
        super().__init__(local_id=local_id, type=Treatment.TYPE, rank=rank, pt=pt, context=context)
        self.treatment_type = treatment_type

    def save(self) -> Tuple:
        return (self.uid, self.treatment_type, self.pt.coordinates, self.context)

    # Treatment step function
    def step(self):
        if Simulation.model.epithelial_barrier_impermeability < Simulation.model.epithelial_barrier_permeability_threshold_start:
            def adjust_bacteria(good_bacteria_factor, pathogenic_bacteria_factor):
                to_add = int((Simulation.params["microbiota_good_bacteria_class"] * np.random.uniform(0, good_bacteria_factor)) / 100)
                Simulation.model.microbiota_good_bacteria_class += to_add
                to_remove = int((Simulation.model.microbiota_pathogenic_bacteria_class * np.random.uniform(0, pathogenic_bacteria_factor)) / 100)
                Simulation.model.microbiota_pathogenic_bacteria_class -= to_remove

            if self.treatment_type == TreatmentType.DIET:
                adjust_bacteria(3, 2)
            elif self.treatment_type == TreatmentType.PROBIOTICS:
                adjust_bacteria(4, 4)