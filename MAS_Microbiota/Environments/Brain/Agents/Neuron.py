from typing import Tuple
from repast4py.space import DiscretePoint as dpt
import numpy as np

from MAS_Microbiota import Simulation
from MAS_Microbiota.Environments import GridAgent


class Neuron(GridAgent):
    TYPE = 7

    def __init__(self, local_id: int, rank: int, initial_state, pt: dpt, context: str):
        super().__init__(local_id=local_id, type=Neuron.TYPE, rank=rank, pt=pt, context=context)
        self.state = initial_state
        self.toRemove = False

    def save(self) -> Tuple:
        return (self.uid, self.state, self.pt.coordinates, self.toRemove, self.context)

    # Neuron step function
    def step(self):
        difference_pro_anti_cytokine = Simulation.model.pro_cytokine - Simulation.model.anti_cytokine
        if difference_pro_anti_cytokine > 0:
            level_of_inflammation = (difference_pro_anti_cytokine * 100) / (Simulation.model.pro_cytokine + Simulation.model.anti_cytokine)
            if np.random.randint(0, 100) < level_of_inflammation:
                self.change_state()
        else:
            pass

    # changes the state of the neuron agent
    def change_state(self):
        if self.state == Simulation.params["neuron_state"]["healthy"]:
            self.state = Simulation.params["neuron_state"]["damaged"]
        elif self.state == Simulation.params["neuron_state"]["damaged"]:
            self.state = Simulation.params["neuron_state"]["dead"]
            self.toRemove = True
            Simulation.model.dead_neuron += 1