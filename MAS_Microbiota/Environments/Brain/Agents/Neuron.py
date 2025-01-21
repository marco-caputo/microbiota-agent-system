from typing import Tuple, Dict, Optional
from repast4py.space import DiscretePoint as dpt
import numpy as np

from MAS_Microbiota import Simulation
from MAS_Microbiota.Environments import GridAgent
from .Precursor import Precursor
from .Neurotransmitter import NeurotransmitterType


class Neuron(GridAgent):
    TYPE = 7

    def __init__(self, local_id: int, rank: int, initial_state, pt: dpt, context: str):
        super().__init__(local_id=local_id, type=Neuron.TYPE, rank=rank, pt=pt, context=context)
        self.state = initial_state
        self.toRemove = False
        # Neurtransmitters the neuron is able to produce given its state
        self.neurotrans_availability = {neuro_type: Simulation.params["neurotrans_initial_availability"]
                                        for neuro_type in NeurotransmitterType}
        self.neurotrans_rate = {neuro_type: 1 for neuro_type in NeurotransmitterType}

    def save(self) -> Tuple:
        return (self.uid, self.state, self.pt.coordinates, self.neurotrans_availability, self.toRemove, self.context)

    # Neuron step function
    def step(self):
        self.check_inflammation()
        self.change_neurotransmitters_to_produce()


    # Changes the state of the neuron agent
    def change_state(self):
        if self.state == Simulation.params["neuron_state"]["healthy"]:
            self.state = Simulation.params["neuron_state"]["damaged"]
        elif self.state == Simulation.params["neuron_state"]["damaged"]:
            self.state = Simulation.params["neuron_state"]["dead"]
            self.toRemove = True
            Simulation.model.dead_neuron += 1


    def check_inflammation(self):
        """
        Checks if the neuron is inflamed and changes its state accordingly.
        """
        difference_pro_anti_cytokine = Simulation.model.pro_cytokine - Simulation.model.anti_cytokine
        if difference_pro_anti_cytokine > 0:
            level_of_inflammation = (difference_pro_anti_cytokine * 100) / (
                    Simulation.model.pro_cytokine + Simulation.model.anti_cytokine)
            if np.random.randint(0, 100) < level_of_inflammation:
                self.change_state()


    def produced_neurotransmitters(self) -> Dict[NeurotransmitterType, int]:
        """
        Returns the neurotransmitters the neuron is able to produce in the current step.
        :return: Dict with neurotransmitter type as key and the amount of neurotransmitter as value
        """
        return {neurotransmitter:
                    min(self.neurotrans_availability[neurotransmitter], self.neurotrans_rate[neurotransmitter])
                    for neurotransmitter in NeurotransmitterType}


    def change_neurotransmitters_to_produce(self):
        """
        Changes the neurotransmitters the neuron will produce in the next step.
        The neuron can increase the availability of a neurotransmitter if it has a precursor in its neighborhood,
        otherwise it can decrease the availability depending on the neuron's state.
        The rate at which the neuron produces neurotransmitters is also changed towards the minimum, which is 1
        per neurotransmitter.
        """
        for neurotransmitter in self.neurotrans_availability:
            self.neurotrans_availability[neurotransmitter] = (max(0,
                self.neurotrans_availability[neurotransmitter] -
                    Simulation.params["neurotrans_decrease"][self.state]*self.neurotrans_rate[neurotransmitter]))

        # Uses a nearby precursor to make more neurotransmitters available
        precursor = self.percept_precursor()
        if precursor is not None:
            neurotrans = np.random.choice(precursor.associated_neurotransmitters())
            self.neurotrans_availability[neurotrans] += Simulation.params["precursor_boost"]

        # Decreases the rate at which the neuron produces neurotransmitters
        for neurotransmitter in self.neurotrans_rate:
            self.neurotrans_rate[neurotransmitter] = max(1, self.neurotrans_rate[neurotransmitter] - 1)



    def percept_precursor(self) -> Optional[Precursor]:
        """
        Returns the precursor agent in the neighborhood of the agent, if any.
        :return: Precursor agent in the neighborhood of the agent
        """
        for ngh_coords in Simulation.model.ngh_finder.find(self.pt.x, self.pt.y):
            nghs_array = Simulation.model.envs['brain'].grid.get_agents(dpt(ngh_coords[0], ngh_coords[1]))
            for ngh in nghs_array:
                if type(ngh) == Precursor:
                    return ngh
        return None
