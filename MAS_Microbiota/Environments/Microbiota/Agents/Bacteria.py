from typing import List
from repast4py import core
from repast4py.space import DiscretePoint as dpt
from repast4py.context import SharedContext as ctx
from MAS_Microbiota import Simulation
import numpy as np, random as rd, uuid

from MAS_Microbiota.Environments import ResourceAgent, GridAgent
from MAS_Microbiota.Environments.Microbiota.Agents import Substrate
from MAS_Microbiota.Environments.Microbiota.Agents import SCFA


# a bacteria only exists within the microbiota, 
# so there might be no need to parameterize the context.
class Bacterium(GridAgent):
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
    def __init__(self, local_id:int, rank:int, context:str, pt:dpt) -> core.Agent:
        super().__init__(id=local_id, type=Bacterium.TYPE , rank=rank, pt=pt, context=context)
        self.TYPE = type
        self.context = context
        self.rank = rank
        self.pt = pt
        self.energy_level = Simulation.params["bacteria_states"][Simulation.params["bacterial_initial_state"]]
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

        Returns
        -------
        tuple[List, List]
            A tuple containing the list of neighbouring Bacteria agents and 
            the list of neighbouring ResourceAgents.
        """
        return self.__find_nghs_bacteria__(), self.__find_nghs_nutrients__()

    # Could rename to quorum_sensing.
    def __find_nghs_bacteria__(self) -> List['Bacterium']:
        """
        Finds the neighbouring Bacteria around this Bacterium within the
        distance of a unit from its current position.

        Returns
        -------
        List[Bacterium]
            A list of neighbouring Bacteria agents.
        """
        nghs_coords = Simulation.model.ngh_finder.find(self.pt.x, self.pt.y) #returns a list of coordinates around this point
        return self.__check_for_nghs_bacteria__(nghs_coords)
    
    # perhaps i can just collapse this method and the check method together.
    def __find_nghs_nutrients__(self) -> List[ResourceAgent]:
        """
        Finds the neighbouring ResourceAgent around this Bacterium within the 
        distance of a unit from its current position.

        returns
        -------
        List[ResourceAgent]
            A list of neighbouring Resource agents.
        """
        nghs_coords = Simulation.model.ngh_finder.find(self.pt.x, self.pt.y) #returns a list of coordinates around this point
        return self.__check_for_nghs_resources__(nghs_coords)
    
    
    def __check_for_nghs_bacteria__(self, nghs_coords:np.NDArray) -> List['Bacterium']:
        """
        Given a list of coordinates around an agent it checks for neighbouring 
        agents which are also bacteria.
        
        Parameters
        ----------
        param nghs_coords: List[dpt]
            list of coordinates around this agent.

        Returns
        -------
        List[Bacterium]
            A list of neighbouring Bacteria agents.
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
        Given a list of coordinates around an agent it checks for neighbouring 
        agents which are resources - namely SCFA or Substrate.
        
        Parameters
        ----------
        param nghs_coords: List[dpt]
            list of coordinates around this agent.

        Returns
        -------
        List[ResourceAgent]
            A list of neighbouring Resource agents.
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
        if self.energy_level == Simulation.params["bacteria_states"]["Maximum"] and len(per[1]) > 0:
            self.fission()
        elif Simulation.params["bacteria_states"]["None"] < self.energy_level < Simulation.params["bacteria_states"]["Maximum"] and any(isinstance(item, Substrate) for item in per[1]): # i should check that the resource is instance of Substrate on the right side of the and operator
            self.ferment(resources=per[1])
        elif self.energy_level < Simulation.params["bacteria_states"]["Maximum"] and any(isinstance(item, SCFA) for item in per[1]):
            self.consume(resources=per[1])
        elif self.energy_level >= Simulation.params["bacteria_states"]["High"] and not any(isinstance(item, SCFA) for item in per[1]):
            self.release_bacteriocins()





    # should we limit fission to the immediate surroundings or 
    # should we allow it to generate in random positions?
    def fission(self) -> None:
        if self.energy_level == Simulation.params["bacteria_states"]["High"]:
            pt = ... #has to be an empty neighbour spot in the grid
            # since id is assigned by the model during agent creation it is unlikely that 
            # we will need to generate a new id for the clone right here, thus unable to 
            # create the clone, unless we share the same id or change the id assigning strategy.

            # we could have uuid generate the id for the clone

            ### check if there is an empty point around this bacterium
            ### if there is, create a clone of the bacterium at that point

            clone = Bacterium(...) #TODO identify the best way to position the clone
            
            ...

    def ferment(self, resources:List[ResourceAgent]) -> None:
        """
        Ferments a Substrate agent into a metabolite agent or 
        produces more energy for the Bacterium, given however that at least
        a Substrate is present in the vicinity.

        Parameters
        ----------
        param resources: List[ResourceAgent]
            A list of ResourceAgent agents in the vicinity of the Bacterium.

        return
        ------
        None
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

        Parameters
        ----------
        param resources: List[ResourceAgent]
            A list of ResourceAgent agents in the vicinity of the Bacterium.

        return
        ------
        None
        """
        if len(resources) > 0:
            index = rd.randint(0, len(resources)-1)
            resource = resources.pop(index)
            resource.toRemove = True
            self.increase_energy()


    def release_bacteriocins(self) -> None:
        """
        Releases bacteriocins into the environment to kill neighbouring competing 
        Bacterium agents.
        
        return
        ------
        None
        """
        # is this an agent too?
        # how many bacteria can a bacteriocin kill before it runs our its life course?
        # how many bacteriocins can a Bacterium release?
        ...


    def idle(self) -> None:
        pass


    
    ###### Utility Methods   ######

    def produce_metabolite(self) -> None:
        """
        Produces a Metabolite agent in the environment.
        """
        ...


    def find_empty_pts(pt:dpt)-> List[dpt]:
        """
        Finds the empty points around a given point.

        Parameters
        ----------
        param pt: dpt
            The point for which we want to find the empty points around.

        return
        ------
        List[dpt]
            A list of empty points around the given point.
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
        Returns the string or integer representation of the internal state 
        of the Bacterium that corresponds to the given value.

        Parameters
        ----------
        param value: int|str
            The value whose correspondent to get in the internal state of the Bacterium.
        
        return
        ------
        int|str
            The string or integer representation of the internal state of the Bacterium.
        """
        codomain = {val: key for key, val in Simulation.params["bacteria_states"].items()}
            
        if type(value) == str:
            return Simulation.params["bacteria_states"][value]
        elif type(value) == int:
            return codomain[value]
    

    def increase_energy(self) -> None:
        """
        Increases the energy level of the Bacterium by one unit and updates its internal state.
        
        return
        ------
        None
        """
        self.energy_level += 1
        self.update_internal_state()

    
    def update_internal_state(self) -> None:
        """
        Updates the current state of the Bacterium in accordance with
        its remaining energy levels.

        return
        ------
        None
        """
        self.energy_level = True if Simulation.params["bacteria_states"]["Maximum"] else self.duplicate = False # if energy level is maximum, the bacteria will fission.




