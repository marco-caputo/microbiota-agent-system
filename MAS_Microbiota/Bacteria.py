from typing import List
from repast4py import core
from repast4py.space import DiscretePoint as dpt
from repast4py.context import SharedContext as ctx
from MAS_Microbiota import Simulation
import numpy as np

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

    def __init__(self, local_id:int, rank:int, context:ctx, pt:dpt) -> core.Agent:
        super().__init__(id=local_id, type=Bacterium.TYPE , rank=rank)
        self.TYPE = type
        self.pt = pt
        self.energy_level = Simulation.params["bacterial_in_state"][Simulation.params["bacterial_initial_state"]]
        self.duplicate = False


    def step(self) -> None:
        """
        This function describes the behaviour of the bacteria agent in the environment.
        
        """


        ...

    ##### THE AGENT'S PERCEPTION
    def percept_surroundings(self) -> None:
        """
        This method observes the surrounding of the Bacterium agent for other Bacteria agents or Nutrient agents.
        """
        #calls @__find_nghs_bacteria__
        #calls @__find_nutrients__
        ...

    # Could rename to quorum_sensing.
    def __find_nghs_bacteria__(self) -> List['Bacterium']:
        """
        Finds the neighbouring Bacteria around this Bacterium within the distance of a unit from its current position
        """
        nghs_coords = Simulation.model.ngh_finder.find(self.pt.x, self.pt.y)
        return self.__check_nghs_bacteria__(nghs_coords)
    

    def __find_nutrients__(self):

        ...
    
    #This is pretty much a copy and paste of the check_oligomer method in Microglia 
    #   but it should work the same.
    # The extra quotes around the return class type hint are used to suppress the warning that would otherwise be present.
    # I think i shouldn't return at the first successful if statement but should gather all nghs and return them at once, 
    #   else the loop breaks prematurely.
    def __check_nghs_bacteria__(nghs_coords:np.NDArray) -> List['Bacterium' | None]:
        """
        Given a list of coordinates around an agent it checks for neighbouring agents which are also bacteria.
        @param nghs_coords: list of coordinates around this agent.
        """
        result = []
        for ngh_coord in nghs_coords:
            #need to define a grid for our microbiota environment to make this work.
            ngh_array = Simulation.model.microbiota_grid.get_agents(dpt(ngh_coord[0], ngh_coord[1]))
            for ngh in ngh_array:
                if type(ngh) == Bacterium:
                    result.append(ngh)
        return result
        ...
    
    ##### THE AGENT'S ACTIONS
    def perform_action(self):
        ...

    def fission(self):
        ...

    def ferment(self):
        ...

    def consume(self):
        ...
    
    def emit_bacteriocins(self):
        ...

    def idle(self):
        ...




