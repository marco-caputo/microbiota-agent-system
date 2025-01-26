from enum import IntEnum
from typing import Tuple
from repast4py.space import DiscretePoint as dpt

from MAS_Microbiota import Simulation
from MAS_Microbiota.Environments import GridAgent
from MAS_Microbiota.Environments.Microbiota.Agents import SubstrateType, Bacterium, EnergyLevel


class ExternalInputType(IntEnum):
    DIET = 1
    ANTIBIOTICS = 2
    STRESS = 3

class ExternalInput(GridAgent):

    TYPE = 4

    def __init__(self, local_id: int, rank: int, input_type: ExternalInputType, pt: dpt, context):
        super().__init__(local_id=local_id, type=ExternalInput.TYPE, rank=rank, pt=pt, context=context)
        self.input_type = input_type
        self.microbiota_env = Simulation.model.envs['microbiota']

    def save(self) -> Tuple:
        return (self.uid, int(self.input_type), self.pt.coordinates, self.context)

    # External input step function
    def step(self):
        if (Simulation.model.epithelial_barrier_impermeability >=
                Simulation.params["epithelial_barrier"]["permeability_threshold_stop"]):
            if self.input_type == ExternalInputType.DIET:
                self._diet_input_action()
            elif self.input_type == ExternalInputType.ANTIBIOTICS:
                self._kill_bacteria(Simulation.params["external_input_antibiotics_factor"])
            else:
                self._boost_pathogenic_bacteria(Simulation.params["external_input_stress_factor"])

    def _diet_input_action(self):
        """
        Modifies the substrates to be introduced in the environment in the next step, representing the effect of
        a diet rich in sugar and poor in fiber.
        """
        self.microbiota_env.substrates_to_add[SubstrateType.SUGAR] += \
            Simulation.params["diet_substrate"]["external_input_influence"]["sugar"]
        self.microbiota_env.substrates_to_add[SubstrateType.CARBOYDRATE] += \
            Simulation.params["diet_substrate"]["external_input_influence"]["carboydrate"]
        self.microbiota_env.substrates_to_add[SubstrateType.FIBER]["fiber"] += \
            Simulation.params["diet_substrate"]["external_input_influence"]["fiber"]

    def _kill_bacteria(self, bacteria_factor):
        """
        Kills a percentage of the bacteria in the environment according to the given factor.
        This method does not make distinction between good and pathogenic bacteria.
        :param bacteria_factor: percentage of bacteria to kill
        """
        to_remove = (
            int(((Simulation.model.microbiota_good_bacteria_count +
                  Simulation.model.microbiota_pathogenic_bacteria_count) *
                         Simulation.model.rng.uniform(0, bacteria_factor)) / 100))
        bacteria = [b for b in self.microbiota_env.context.agents() if isinstance(b, Bacterium)]
        for b in Simulation.model.rng.sample(bacteria, to_remove):
            b.toRemove = True

    def _boost_pathogenic_bacteria(self, bacteria_factor):
        """
        Boosts a percentage of the pathogenic bacteria in the environment according to the given factor.
        The boosts consists in setting the energy level of the bacteria to the maximum.
        :param bacteria_factor: percentage of pathogenic bacteria to boost
        """
        to_boost = int((Simulation.params["microbiota_pathogenic_bacteria_class"] *
                      Simulation.model.rng.uniform(0, bacteria_factor)) / 100)
        pathogenic_bacteria = [b for b in self.microbiota_env.context.agents() if
                               isinstance(b, Bacterium) and b.causes_inflammation()]
        if len(pathogenic_bacteria) > 0:
            for _ in range(to_boost):
                b = Simulation.model.rng.choice(pathogenic_bacteria)
                b.enegy_level = EnergyLevel.MAXIMUM