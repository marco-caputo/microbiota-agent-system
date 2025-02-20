from enum import IntEnum
from typing import Tuple
from repast4py.space import DiscretePoint as dpt
import numpy as np

from MAS_Microbiota import Simulation
from MAS_Microbiota.Environments import GridAgent
from MAS_Microbiota.Environments.Microbiota.Agents import SubstrateType, Bifidobacteriaceae, Lachnospiraceae


class TreatmentType(IntEnum):
    DIET = 1
    PROBIOTICS = 2

class Treatment(GridAgent):

    TYPE = 5

    PROBIOTICS_BACTERIA = [Bifidobacteriaceae, Lachnospiraceae]

    def __init__(self, local_id: int, rank: int, treatment_type: TreatmentType, pt: dpt, context):
        super().__init__(local_id=local_id, type=Treatment.TYPE, rank=rank, pt=pt, context=context)
        self.treatment_type = treatment_type

    def save(self) -> Tuple:
        return (self.uid, int(self.treatment_type), self.pt.coordinates, self.context)

    # Treatment step function
    def step(self):
        if Simulation.model.epithelial_barrier_impermeability < Simulation.model.epithelial_barrier_permeability_threshold_start:

            if self.treatment_type == TreatmentType.DIET:
                self._diet_input_action()
            elif self.treatment_type == TreatmentType.PROBIOTICS:
                self._introduce_good_bacteria(Simulation.params["treatment_probiotics_factor"])

    def _diet_input_action(self):
        """
        Modifies the substrates to be introduced in the environment in the next step, representing the effect of
        a diet rich in fiber and poor in sugar.
        """
        for type in list(SubstrateType):
            Simulation.model.envs['microbiota'].substrates_to_add[type] += \
                Simulation.params["diet_substrates"]["treatment_influence"][type.name.lower()]

    def _introduce_good_bacteria(self, probiotics_factor):
        """
        Introduces good bacteria in the environment according to the given factor.
        :param probiotics_factor: percentage of good bacteria to introduce
        """
        to_add = int((Simulation.model.microbiota_good_bacteria_count *
                      Simulation.model.rng.uniform(0, probiotics_factor)) / 100)
        for _ in range(to_add):
            bacteria_class = Simulation.model.rng.choice(self.PROBIOTICS_BACTERIA)
            random_pt = Simulation.model.envs['microbiota'].grid.get_random_local_pt(Simulation.model.rng)
            bacterium_to_add = bacteria_class(Simulation.model.new_id(), Simulation.model.rank, random_pt, self.context)
            Simulation.model.envs['microbiota'].bacteria_to_add.append(bacterium_to_add)