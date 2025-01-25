from abc import abstractmethod
from typing import List, Type, Dict
from repast4py.space import DiscretePoint as dpt
from MAS_Microbiota import Simulation
import numpy as np
from enum import IntEnum

from MAS_Microbiota.Environments import ResourceAgent, GridAgent
from MAS_Microbiota.Environments.Brain.Agents.Precursor import PrecursorType, Precursor
from MAS_Microbiota.Environments.Microbiota.Agents import Substrate
from MAS_Microbiota.Environments.Microbiota.Agents import SCFA
from MAS_Microbiota.Environments.Microbiota.Agents.SCFA import SCFAType
from MAS_Microbiota.Environments.Microbiota.Agents.Substrate import SubstrateType


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

    pt: dpt
    energy_level: EnergyLevel
    toFission: bool
    toFerment: Dict[Type[ResourceAgent], bool]

    def __init__(self, local_id: int, rank: int, pt: dpt, context: str):
        super().__init__(local_id=local_id, type=Bacterium.TYPE, rank=rank, pt=pt, context=context)
        self.rank = rank
        self.pt = pt
        self.energy_level = EnergyLevel[Simulation.params["bacterial_initial_state"]]
        self.toRemove: bool = False
        self.toFission: bool = False
        self.toFerment: Dict[Type[ResourceAgent], bool] = {
            Substrate: False,
            Precursor: False
        }
        self.fermentedPrecursor: int = 0

    def save(self):
        return (self.uid, type(self), self.pt.coordinates, self.context, self.toFission, self.toFerment[Substrate],
                self.toFerment[Precursor], self.fermentedPrecursor, self.toRemove, int(self.energy_level))

    def step(self) -> None:
        """
        This function describes the default behaviour of the bacteria agent in the environment.
        """
        per = self.percept()
        self.perform_action(*per)

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
                if type(ngh) in [SCFA, Substrate, Precursor] and not ngh.toRemove:
                    toAdd = False
                    if type(ngh) == SCFA:
                        toAdd = SCFAType in self.consumable_scfa()
                    elif type(ngh) == Substrate:
                        toAdd = SubstrateType in self.fermentable_substrates()
                    elif type(ngh) == Precursor:
                        toAdd = PrecursorType in self.fermentable_precursors()
                    if toAdd:
                        result.append(ngh)
        return result

    ##### THE AGENT'S ACTIONS
    #type of perception and return type to be decided
    # perception might just be called inside the method instead of being passed as an argument.
    def perform_action(self, percieved_bacteria: List['Bacterium'], percieved_resources: List[ResourceAgent]) -> None:
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
        if self.energy_level == EnergyLevel.MAXIMUM and len(percieved_resources) > 0:
            self.fission()

        elif (self.can_ferment_substrates() and
              (EnergyLevel.NONE < self.energy_level < EnergyLevel.MAXIMUM) and
              any(isinstance(item, Substrate) for item in percieved_resources)):
            self.ferment(Substrate, percieved_resources)

        elif (self.can_ferment_precursors() and
                (EnergyLevel.NONE < self.energy_level < EnergyLevel.MAXIMUM) and
                any(isinstance(item, Precursor) for item in percieved_resources)):
            self.ferment(Precursor, percieved_resources)

        elif self.energy_level < EnergyLevel.MAXIMUM and any(isinstance(item, SCFA) for item in percieved_resources):
            self.consume(percieved_resources)

        elif (self.can_move() and self.energy_level > EnergyLevel.NONE and len(percieved_resources) == 0
                and len(Simulation.model.envs['microbiota'].find_bact_free_nghs(self.pt)) != 0):
            self.move()

        elif (self.can_release_bacteriocins() and
              self.energy_level >= EnergyLevel.HIGH and len(percieved_resources) == 0):
            self.release_bacteriocins(percieved_bacteria)

        else:
            self.idle()

    def fission(self) -> None:
        self.toFission = True
        self.update_energy(Simulation.params["bacteria_energy_deltas"]["fission"])

    def ferment(self, fermentable_type: type[ResourceAgent], resources: List[ResourceAgent]) -> None:
        """
        Ferments a resource agent into a metabolite agent and
        produces more energy for the Bacterium, given however that at least
        a fermentable resource agent is present in the vicinity for this Bacterium.

        :param fermentable_type: The type of fermentable resource agent.
        :param resources: The list of ResourceAgent agents in the vicinity of the Bacterium.
        """
        fermentable_resources = [resource for resource in resources if isinstance(resource, fermentable_type)]
        if len(fermentable_resources) > 0:
            nutrient = Simulation.model.rng.choice(fermentable_resources)
            nutrient.toRemove = True
            self.update_energy(Simulation.params["bacteria_energy_deltas"]["ferment"])
            self.toFerment[fermentable_type] = True
            if fermentable_type == Precursor:
                self.fermentedPrecursor = int(nutrient.precursor_type)

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

    def move(self) -> None:
        """
        Moves the Bacterium agent to a random neighbouring position free of other Bacteria agents.
        """
        free_nghs = Simulation.model.envs['microbiota'].find_bact_free_nghs(self.pt)
        if len(free_nghs) > 0:
            rand_pos = Simulation.model.rng.choice(free_nghs)
            chosen_dpt = dpt(rand_pos[0], rand_pos[1])
            Simulation.model.move(self, chosen_dpt, self.context)
            self.update_energy(Simulation.params["bacteria_energy_deltas"]["move"])

    def idle(self) -> None:
        # Energy level is reduced by 1 unit for idling.
        self.update_energy(Simulation.params["bacteria_energy_deltas"]["idle"])

    def update_energy(self, delta: int) -> None:
        """
        Increases or decreases the energy level of the Bacterium according to the given integer value
        and updates its internal state.
        If the energy level is below the minimum level, the Bacterium is marked for removal, indicating it is dead.

        :param delta: The integer value to increase or decrease the energy level
        """
        if int(self.energy_level) + delta < int(EnergyLevel.NONE):
            self.toRemove = True
        self.energy_level = EnergyLevel(
            max(int(EnergyLevel.NONE),
                min(int(EnergyLevel.MAXIMUM),
                    int(self.energy_level) + delta)
                )
        )

    def can_ferment_substrates(self) -> bool:
        """
        Returns True if the bacterium can ferment a substrate.
        """
        return len(self.fermentable_substrates()) > 0

    def can_ferment_precursors(self) -> bool:
        """
        Returns True if the bacterium can ferment a precursor.
        """
        return len(self.fermentable_precursors()) > 0

    @abstractmethod
    def fermentable_substrates(self) -> List[SubstrateType]:
        """
        Returns the list of substrates types fermentable by this bacterium.
        """
        ...

    @abstractmethod
    def fermentable_precursors(self) -> List[PrecursorType]:
        """
        Returns the list of precursors fermentable by this bacterium.
        """
        ...

    @abstractmethod
    def consumable_scfa(self) -> List[SCFAType]:
        """
        Returns the list of SCFA types consumable by this bacterium.
        """
        ...

    @abstractmethod
    def produced_scfa(self) -> List[SCFAType]:
        """
        Returns the list of SCFAs produced.
        """
        ...

    @abstractmethod
    def produced_precursors(self) -> List[PrecursorType]:
        """
        Returns the list of Precursors produced.
        """
        ...

    @abstractmethod
    def can_release_bacteriocins(self) -> bool:
        """
        Returns True if the bacterium can release bacteriocins.
        """
        ...

    @abstractmethod
    def can_move(self) -> bool:
        """
        Returns True if the bacterium can move.
        """
        return True
