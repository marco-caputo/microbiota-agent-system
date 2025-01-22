from typing import List
from repast4py import core
from repast4py.space import DiscretePoint as dpt
from repast4py.context import SharedContext as ctx
from MAS_Microbiota import Simulation
import numpy as np, random as rd, uuid

from MAS_Microbiota.Environments import ResourceAgent
from MAS_Microbiota.Environments.Microbiota.Agents import Substrate
from MAS_Microbiota.Environments.Microbiota.Agents import SCFA


# a bacteria only exists within the microbiota, 
# so there might be no need to parameterize the context.
class Bacterium(core.Agent):
    """
    This class represents a Bacteria...
    @param TYPE
    @param internal_states
    @param em
    
    """
    # if type is a parameter variable or an initialized constant still needs to be decided.
    TYPE: int = 99

    energy_level:int
    duplicate:bool
    pt:dpt

    #we could import uuid and use it to generate our ids instead of local_id input
    def __init__(self, rank:int, context:ctx, pt:dpt) -> core.Agent:
        super().__init__(id=uuid.uuid4(), type=Bacterium.TYPE , rank=rank)
        self.TYPE = type
        self.context = context
        self.rank = rank
        self.pt = pt
        self.energy_level = Simulation.params["bacterial_in_state"][Simulation.params["bacterial_initial_state"]]
        self.duplicate = False


    def step(self) -> None:
        """
        This function describes the behaviour of the bacteria agent in the environment.
        
        """


        ...

    ##### THE AGENT'S PERCEPTION
    def percept(self) -> tuple[List, List]:
        """
        This method observes the surrounding of the Bacterium agent for other 
        Bacteria agents or Nutrient agents.
        """
        return self.__find_nghs_bacteria__(), self.__find_nghs_nutrients__()

    # Could rename to quorum_sensing.
    def __find_nghs_bacteria__(self) -> List['Bacterium']:
        """
        Finds the neighbouring Bacteria around this Bacterium within the
        distance of a unit from its current position.
        """
        nghs_coords = Simulation.model.ngh_finder.find(self.pt.x, self.pt.y) #returns a list of coordinates around this point
        return self.__check_for_nghs_bacteria__(nghs_coords)
    
    # perhaps i can just collapse this method and the check method together.
    def __find_nghs_nutrients__(self) -> List[ResourceAgent]:
        """
        Finds the neighbouring ResourceAgent around this Bacterium within the 
        distance of a unit from its current position.
        """
        nghs_coords = Simulation.model.ngh_finder.find(self.pt.x, self.pt.y) #returns a list of coordinates around this point
        return self.__check_for_nghs_resources__(nghs_coords)
    
    
    def __check_for_nghs_bacteria__(self, nghs_coords:np.NDArray) -> List['Bacterium']:
        """
        Given a list of coordinates around an agent it checks for neighbouring agents which are also bacteria.
        
        Parameters
        ----------
        param nghs_coords: List[dpt]
            list of coordinates around this agent.
        """
        result = []
        for ngh_coord in nghs_coords:
            #need to define a grid for our microbiota environment to make this work.
            ngh_array = Simulation.model.microbiota_grid.get_agents(dpt(ngh_coord[0], ngh_coord[1]))
            for ngh in ngh_array:
                if type(ngh) == Bacterium:
                    result.append(ngh)
        return result
    

    def __check_for_nghs_resources__(self, nghs_coords:np.NDArray) -> List[ResourceAgent]:
        """
        Given a list of coordinates around an agent it checks for neighbouring agents which are resources - namely SCFA or Substrate.
        
        Parameters
        ----------
        param nghs_coords: List[dpt]
            list of coordinates around this agent.
        """
        result = []
        for ngh_coord in nghs_coords:
            #need to define a grid for our microbiota environment to make this work.
            ngh_array = Simulation.model.microbiota_grid.get_agents(dpt(ngh_coord[0], ngh_coord[1]))
            for ngh in ngh_array:
                if type(ngh) == ResourceAgent:
                    result.append(ngh)
        return result


    
    ##### THE AGENT'S ACTIONS
    #type of perception and return type to be decided
    # perception might just be called inside the method instead of being passed as an argument.
    def perform_action(self, per:tuple[List, List]) -> None:
        """
        Performs an action on the bacterium based on its inner state and 
        the conditions of its surroundings.

        Parameters
        ----------
        param per: tuple[List, List]
            A tuple containing the list of neighbouring Bacteria agents and 
            the list of neighbouring ResourceAgents.

        """
        if self.energy_level == Simulation.params["bacterial_in_state"]["Maximum"] and len(per[1]) > 0:
            self.fission()
        elif Simulation.params["bacterial_in_state"]["None"] < self.energy_level < Simulation.params["bacterial_in_state"]["Maximum"] and any(isinstance(item, Substrate) for item in per[1]): # i should check that the resource is instance of Substrate on the right side of the and operator
            self.ferment(resources=per[1])
        elif self.energy_level < Simulation.params["bacterial_in_state"]["Maximum"] and any(isinstance(item, SCFA) for item in per[1]):
            self.consume(resources=per[1])
        elif self.energy_level >= Simulation.params["bacterial_in_state"]["High"] and not any(isinstance(item, SCFA) for item in per[1]):
            self.release_bacteriocins()





    # should we limit fission to the immediate surroundings or 
    # should we allow it to generate in random positions?
    def fission(self) -> None:
        if self.energy_level == Simulation.params["bacteria_initernal_state"]["High"]:
            pt = ... #has to be an empty neighbour spot in the grid
            clone = Bacterium(rank=self.rank, context=self.context, pt=pt) #TODO identify the best way to position the clone
            
            ...

    def ferment(self, resources:List[ResourceAgent]) -> None:
        """
        Ferments a Substrate agent into a metabolite agent or 
        produces more energy for the Bacterium, given however that at least
        a Substrate is present in the vicinity.
        """
        if len(resources) > 0:
            for resource in resources:
                if type(resource) == Substrate: 
                    resource.toRemove = True
                    behaviours = [self.increase_energy, self.produce_metabolite]
                    choice_index = rd.randint(0, len(behaviours)-1)
                    behaviours[choice_index]()
                    return
        else: # do nothing
            pass


    def consume(self, resources:List[ResourceAgent]) -> None:
        """
        Consumes a SCFA agent to produce energy for the bacteria.
        """
        if len(resources) > 0:
            index = rd.randint(0, len(resources)-1)
            resource = resources.pop(index)
            resource.toRemove = True
            self.energy_level+=1
        self.update_internal_state()


    def release_bacteriocins(self) -> None:
        """
        Releases bacteriocins into the environment to kill neighbouring competing 
        Bacterium agents."""
        # is this an agent too?
        # how many bacteria can a bacteriocin kill before it runs our its life course?
        # how many bacteriocins can a Bacterium release?
        ...

    def idle(self) -> None:
        pass


    
    ###### Utility Methods   ######

    def find_empty_pts(pt:dpt)-> List[dpt]:
        """
        Finds the empty points around a given point and returns them.
        """
        empty_pt_list = []
        nghs_coords = Simulation.model.ngh_finder.find(pt)
        for ngh_coords in nghs_coords:
            nghs_agents = list(Simulation.model.envs['microbiota'].grid.get_agents(dpt(ngh_coords[0], ngh_coords[1])))
            if len(nghs_agents) == 0:
                 empty_pt_list.append(empty_pt_list)
        return empty_pt_list
    
    def get_internal_state(value:int|str) -> int|str:
        """
        Returns the string or integer representation of the internal state of the Bacterium.
        """
        codomain = {key: val for key, val in Simulation.params["bacterial_in_state"].items()}
            
        if type(value) == str:
            return Simulation.params["bacterial_state"][value]
        elif type(value) == int:
            return codomain[value]
    

    def increase_energy(self) -> None:
        """
        Increases the energy level of the Bacterium by one unit.
        """
        self.energy_level += 1
        self.update_internal_state()

    def produce_metabolite(self) -> None:
        """
        Produces a Metabolite agent in the environment.
        """
        ...

    
    def update_internal_state(self):
        """
        Updates the current state of the Bacterium in accordance with
        its remaining energy levels.
        """
        self.energy_level = True if Simulation.params["bacterial_in_state"]["Maximum"] else self.duplicate = False # if energy level is maximum, the bacteria will fission.




