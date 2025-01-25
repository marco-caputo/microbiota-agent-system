from abc import abstractmethod
from typing import List
from repast4py.space import DiscretePoint as dpt
from MAS_Microbiota import Simulation
import numpy as np
from enum import IntEnum

from MAS_Microbiota.Environments import ResourceAgent, GridAgent
from MAS_Microbiota.Environments.Brain.Agents.Neurotransmitter import NeurotransmitterType
from MAS_Microbiota.Environments.Brain.Agents.Precursor import PrecursorType
from MAS_Microbiota.Environments.Microbiota.Agents import Substrate
from MAS_Microbiota.Environments.Microbiota.Agents import SCFA
from MAS_Microbiota.Environments.Microbiota.Agents.SCFA import SCFAType


class EnergyLevel(IntEnum):
    """
    This class represents the energy level of the Bacterium agent.
    """
    NONE = 0
    VERYLOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERYHIGH = 5
    MAXIMUM = 6


# a bacteria only exists within the microbiota, 
# so there might be no need to parameterize the context.
class Bacterium(GridAgent):
    """
    This class represents a generic unit of Bacteria.
    Subclasses of this class will represent different families of Bacteria.
    """
    # if type is a parameter variable or an initialized constant still needs to be decided.
    TYPE: int = 99

    energy_level: EnergyLevel
    duplicate: bool
    pt: dpt

    def __init__(self, local_id: int, rank: int, context: str, pt: dpt, toFission: bool = False,
                 toFerment: bool = False):
        super().__init__(local_id=local_id, type=Bacterium.TYPE, rank=rank, pt=pt, context=context)
        self.rank = rank
        self.pt = pt
        self.energy_level = EnergyLevel[Simulation.params["bacterial_initial_state"]]
        self.duplicate = False
        self.toFission = toFission
        self.toFerment = toFerment

    @abstractmethod
    def save(self):
        ...

    @abstractmethod
    def produced_scfa(self) -> List[SCFAType]:
        """
        Returns the list of SCFAs produced.
        """
        ...

    @abstractmethod
    def produced_neurotransmitters(self) -> List[NeurotransmitterType]:
        """
        Returns the list of Neurotransmitters produced.
        """
        ...

    @abstractmethod
    def produced_precursors(self) -> List[PrecursorType]:
        """
        Returns the list of Precursors produced.
        """
        ...

    def step(self) -> None:
        """
        This function describes the default behaviour of the bacteria agent in the environment.
        """
        per = self.percept()
        self.perform_action(per)


    ##### THE AGENT'S PERCEPTION
    def percept(self) -> tuple[List, List]:
        """
        This method observes the surrounding of the Bacterium agent for other 
        Bacteria agents or SCFA and Substrate agents.
        Note that only agent not marked for removal are considered.

        :returns: A tuple containing the list of neighbouring Bacteria agents and the list of neighbouring
        SCFA and Substrates.
        """
        nghs_coords = Simulation.model.ngh_finder.find(self.pt.x, self.pt.y)
        return self.__check_for_nghs_bacteria__(nghs_coords), self.__check_for_nghs_resources__(nghs_coords)

    def __check_for_nghs_bacteria__(self, nghs_coords: np.NDArray) -> List['Bacterium']:
        """
        Given a list of coordinates around an agent it checks for neighbouring 
        agents which are also bacteria not marked for removal.

        :param nghs_coords: list of coordinates around this agent.
        :returns: A list of neighbouring Bacteria agents not marked for removal.
        """
        result = []
        for ngh_coord in nghs_coords:
            ngh_array = Simulation.model.envs['microbiota'].grid.get_agents(dpt(ngh_coord[0], ngh_coord[1]))
            for ngh in ngh_array:
                if type(ngh) == Bacterium and not ngh.toRemove:
                    result.append(ngh)
        return result

    def __check_for_nghs_resources__(self, nghs_coords: np.NDArray) -> List[ResourceAgent]:
        """
        Given a list of coordinates around an agent it checks for neighbouring 
        agents which are resources of interest, namely SCFA or Substrate. The
        agents must not be marked for removal.

        :param nghs_coords: list of coordinates around this agent.
        :returns: A list of neighbouring Resource agents not marked for removal.
        """
        result = []
        for ngh_coord in nghs_coords:
            ngh_array = Simulation.model.envs['microbiota'].grid.get_agents(dpt(ngh_coord[0], ngh_coord[1]))
            for ngh in ngh_array:
                if type(ngh) in [SCFA, Substrate] and not ngh.toRemove:
                    result.append(ngh)
        return result

    ##### THE AGENT'S ACTIONS
    #type of perception and return type to be decided
    # perception might just be called inside the method instead of being passed as an argument.
    def perform_action(self, per: tuple[List['Bacterium'], List[ResourceAgent]]) -> None:
        """
        Performs an action on the bacterium based on its inner state and 
        the conditions surrounds it.

        Parameters
        ----------
        param per: tuple[List, List]
            A tuple containing the list of neighbouring Bacteria agents and 
            the list of neighbouring ResourceAgents.

        return
        ------
        None
        """
        if self.energy_level == EnergyLevel.MAXIMUM and len(per[1]) > 0:
            self.fission()
        elif (EnergyLevel.NONE < self.energy_level < EnergyLevel.MAXIMUM and
              any(isinstance(item, Substrate) for item in per[1])):
            self.ferment(resources=per[1])
        elif self.energy_level < EnergyLevel.MAXIMUM and any(isinstance(item, SCFA) for item in per[1]):
            self.consume(resources=per[1])
        elif self.energy_level >= EnergyLevel.HIGH and len(per[1]) == 0:
            self.release_bacteriocins(neighbors=per[0])
        else:
            self.idle()

    def fission(self) -> None:
        self.toFission = True
        self.update_energy(Simulation.params["bacteria_energy_deltas"]["fission"])

    def ferment(self, resources: List[ResourceAgent]) -> None:
        """
        Ferments a Substrate agent into a metabolite agent or 
        produces more energy for the Bacterium, given however that at least
        a Substrate is present in the vicinity.

        Parameters
        ----------
        param resources: List[ResourceAgent]
            A list of ResourceAgent agents in the vicinity of the bacterium.

        return
        ------
        None
        """
        substrates = [resource for resource in resources if isinstance(resource, Substrate)]
        if len(substrates) > 0:
            substrate = Simulation.model.rng.choice(substrates)
            substrate.toRemove = True
            self.update_energy(Simulation.params["bacteria_energy_deltas"]["ferment"])
            self.toFerment = True

    def consume(self, resources: List[ResourceAgent]) -> None:
        """
        Consumes a SCFA agent to produce energy for the bacterium.

        Parameters
        ----------
        param resources: List[ResourceAgent]
            A list of ResourceAgent agents in the vicinity of the Bacterium.

        return
        ------
        None
        """
        if len(resources) > 0:
            resource = Simulation.model.rng.choice(resources)
            resource.toRemove = True
            self.update_energy(Simulation.params["bacteria_energy_deltas"]["consume"])

    def release_bacteriocins(self, neighbors: List['Bacterium']) -> None:
        """
        Releases bacteriocins into the environment to kill neighbouring competing 
        Bacterium agents. This is performed by setting the toRemove attribute of
        the neighbouring Bacterium agents to True.

        :param neighbors: A list of neighbouring Bacterium agents.
        """
        for neighbor in neighbors:
            neighbor.toRemove = True
        self.update_energy(Simulation.params["bacteria_energy_deltas"]["bacteriocins"])

    def idle(self) -> None:
        # Energy level is reduced by 1 unit for idling.
        self.update_energy(Simulation.params["bacteria_energy_deltas"]["idle"])

    def update_energy(self, delta: int) -> None:
        """
        Increases or decreases the energy level of the Bacterium according to the given integer value
        and updates its internal state.

        :param delta: The integer value to increase or decrease the energy level
        """
        self.energy_level = EnergyLevel(
            max(int(EnergyLevel.NONE),
                min(int(EnergyLevel.MAXIMUM),
                    int(self.energy_level) + delta)
                )
        )
