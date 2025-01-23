from MAS_Microbiota import Simulation, restore_agent
from MAS_Microbiota.Environments import GridAgent, GridEnvironment
from MAS_Microbiota.Environments.Brain.Agents.Neurotransmitter import Neurotransmitter
from MAS_Microbiota.Environments.Brain.Agents.Precursor import Precursor
from MAS_Microbiota.Environments.Microbiota.Agents.Bacteria import Bacterium
from MAS_Microbiota.Environments.Microbiota.Agents.Metabolite import Metabolite
from MAS_Microbiota.Environments.Microbiota.Agents.SCFA import SCFAType, SCFA
from MAS_Microbiota.Environments.Microbiota.Agents.Substrate import Substrate
import numpy as np

class Microbiota(GridEnvironment):

    def __init__(self, context: GridEnvironment, grid: GridAgent):
        super().__init__(context, grid)

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
            neighbours = Simulation.model.ngh_finder.find(bacterium.point.x, bacterium.point.y) # Neighbours of the bacterium...
            if bacterium.to_fission:
                self.fission(bacterium, neighbours)
            elif bacterium.to_ferment:
                self.ferment(bacterium, neighbours)

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

    @staticmethod
    def agent_types():
        return [
            ('bacterium.count', Bacterium, None),
            ('metabolite.count', Metabolite, None),
            ('scfa.count', SCFAType, None),
            ('substrate.count', Substrate, None)
        ]

    def agents_to_remove(self):
        return Bacterium, Metabolite, SCFAType, Substrate

    def fission(self, bacterium, neighbours):
        point = neighbours.get_random_local_pt(Simulation.model.rng)
        Simulation.model.added_agents_id += 1
        new_bacterium = Bacterium(Simulation.model.added_agents_id, Simulation.model.rank, self.context, point)
        self.context.agents().add(new_bacterium)
        bacterium.to_fission = False

    def ferment(self, bacterium, neighbours):
        point = neighbours.get_random_local_pt(Simulation.model.rng)
        # modify here: according to the bacterium's family, different SCFAs are created.
        Simulation.model.added_agents_id += 1
        self.add_metabolite(bacterium.produced_scfa(), SCFA, point)
        self.add_metabolite(bacterium.produced_precursors(), Precursor, point)
        self.add_metabolite(bacterium.produced_neurotransmitters(), Neurotransmitter, point)

    """
    Based on the assumption that all agents have the same constructor signature.
    """
    def add_metabolite(self, types, agent_class, point):
        if len(types) > 0:
            current_type = np.random.choice(types)
            Simulation.model.added_agents_id += 1
            agent = agent_class(Simulation.model.added_agents_id, Simulation.model.rank, current_type, point, self.context)
            self.context.agents().add(agent)
