import numpy as np
from repast4py.space import DiscretePoint as dpt

from MAS_Microbiota import Simulation, restore_agent
from MAS_Microbiota.Environments import GridAgent, GridEnvironment, ResourceAgent
from MAS_Microbiota.Environments.Brain.Agents.Neurotransmitter import Neurotransmitter
from MAS_Microbiota.Environments.Brain.Agents.Precursor import Precursor, PrecursorType
from MAS_Microbiota.Environments.Microbiota.Agents.Bacterium import Bacterium
from MAS_Microbiota.Environments.Microbiota.Agents.Metabolite import Metabolite
from MAS_Microbiota.Environments.Microbiota.Agents.SCFA import SCFAType, SCFA
from MAS_Microbiota.Environments.Microbiota.Agents.Substrate import Substrate

class Microbiota(GridEnvironment):

    def __init__(self, context: GridEnvironment, grid: GridAgent):
        super().__init__(context, grid)

    @staticmethod
    def agent_types():
        return [
            ('bacterium.count', Bacterium, None),
            ('scfa.count', SCFAType, None),
            ('substrate.count', Substrate, None),
            ('precursor.count', Precursor, None),
            ('neurotransmitter.count', Neurotransmitter, None)
        ]

    def step(self):
        removed_ids = set()
        self.context.synchronize(restore_agent)
        self.remove_agents(removed_ids)
        self.make_agents_steps()
        self.apply_actions()
        self.context.synchronize(restore_agent)

    def apply_actions(self):
        for bacterium in (agent for agent in self.context.agents() if isinstance(agent, Bacterium)): # For each bacterium in the context...
            bacterium.step() # Call the step method of the bacterium.
            if bacterium.toFission:
                self.fission(bacterium)
            for fermentable_type in bacterium.toFerment:
                if bacterium.toFerment[fermentable_type]:
                    self.ferment(bacterium, fermentable_type)

    """
    def action(self):
        bacteria = []
        neighbours = [] # List of neighbours for each bacterium.
        for agent in self.context.agents(): # For each agent in the context...
            if isinstance(agent, Bacterium): # If the agent is a bacterium...
                bacteria.append(agent) # Add the bacterium to the list of bacteria.
                neighbours_coordinates = Simulation.model.ngh_finder.find(agent.point.x, agent.point.y) # Get bacterium's neighbours' coordinates.
                for neighbour_coordinates in neighbours_coordinates: # For each neighbour's coordinates...
                    neighbours_array = Simulation.model.gut_grid.get_agents(dpt(neighbour_coordinates[0], neighbour_coordinates[1])) # Get the agents in the neighbour.
                    for neighbour in neighbours_array:
                        neighbours.append(neighbour)
        self.perform_action(bacteria, neighbours)
    """

    def agents_to_remove(self):
        return Bacterium, SCFAType, Substrate, Precursor, Neurotransmitter

    def fission(self, bacterium):
        """
        Applies fission to a bacterium agent, creating a new bacterium agent in an empty position around the bacterium.
        If no empty position is found, the fission is not applied, but the state of the bacterium remains unchanged
        in order to try again in the next step.

        :param bacterium: The bacterium agent to apply fission
        """
        empty_ngh_pts = self.find_bact_free_nghs(bacterium.pt)
        if len(empty_ngh_pts) == 0:
            return
        point = Simulation.model.rng.choice(empty_ngh_pts)
        bact_class = type(bacterium)
        new_bacterium = bact_class(Simulation.model.new_id(), Simulation.model.rank, self.context, point)
        self.context.agents().add(new_bacterium)
        bacterium.toFission = False

    def find_bact_free_nghs(self, pt: dpt) -> list[dpt]:
        """
        Finds the position around a given point that do not have any bacterium agent.
        :param pt: The point to check
        :return: A list of empty points
        """
        empty_pt_list = []
        nghs_coords = Simulation.model.ngh_finder.find(pt)
        for ngh_coords in nghs_coords:
            nghs_agents = list(self.grid.get_agents(dpt(ngh_coords[0], ngh_coords[1])))
            if len([ag for ag in nghs_agents if isinstance(ag, Bacterium)]) == 0:
                 empty_pt_list.append(empty_pt_list)
        return empty_pt_list

    def ferment(self, bacterium: Bacterium, fermentable_type: type[ResourceAgent]):
        neighbours = Simulation.model.ngh_finder.find(bacterium.point.x, bacterium.point.y)  # Neighbours of the bacterium...
        point = neighbours.get_random_local_pt(Simulation.model.rng)
        if fermentable_type == Substrate:
            self.add_metabolite(bacterium.produced_scfa(), SCFA, point)
            self.add_metabolite(bacterium.produced_precursors(), Precursor, point)
        elif fermentable_type == Precursor and bacterium.fermentedPrecursor != 0:
            fermented_precursor_type = PrecursorType(bacterium.fermentedPrecursor)
            self.add_metabolite(fermented_precursor_type.associated_neurotransmitters(), Neurotransmitter, point)
            bacterium.fermentedPrecursor = 0


    # Based on the assumption that all metabolite and neurotransmitters agents have the same constructor signature.
    def add_metabolite(self, types, agent_class, point):
        if len(types) > 0:
            current_type = Simulation.model.rng.choice(types)
            agent = agent_class(Simulation.model.new_id(), Simulation.model.rank, current_type, point, self.context)
            self.context.agents().add(agent)
