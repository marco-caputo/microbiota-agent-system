from typing import Tuple
from repast4py import core
from repast4py.space import DiscretePoint as dpt
import numpy as np

from MAS_Microbiota import Simulation

class Treatment(core.Agent):

    TYPE = 5

    def __init__(self, local_id: int, rank: int, pt: dpt, context):
        super().__init__(id=local_id, type=Treatment.TYPE, rank=rank)
        possible_types = [Simulation.params["treatment_input"]["diet"],Simulation.params["treatment_input"]["probiotics"]]
        random_index = np.random.randint(0, len(possible_types))
        input_name = possible_types[random_index]
        self.pt = pt
        self.input_name = input_name
        self.context = context

    def save(self) -> Tuple:
        return (self.uid, self.input_name, self.pt.coordinates, self.context)

    # Treatment step function
    def step(self):
        if Simulation.model.barrier_impermeability < Simulation.model.barrier_permeability_threshold_start:
            def adjust_bacteria(good_bacteria_factor, pathogenic_bacteria_factor):
                to_add = int((Simulation.params["microbiota_good_bacteria_class"] * np.random.uniform(0, good_bacteria_factor)) / 100)
                Simulation.model.microbiota_good_bacteria_class += to_add
                to_remove = int((Simulation.model.microbiota_pathogenic_bacteria_class * np.random.uniform(0, pathogenic_bacteria_factor)) / 100)
                Simulation.model.microbiota_pathogenic_bacteria_class -= to_remove

            if self.input_name == Simulation.params["treatment_input"]["diet"]:
                adjust_bacteria(3, 2)
            elif self.input_name == Simulation.params["treatment_input"]["probiotics"]:
                adjust_bacteria(4, 4)