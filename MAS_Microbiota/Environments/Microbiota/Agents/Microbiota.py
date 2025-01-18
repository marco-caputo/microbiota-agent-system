from Bacterium import Energy
from MAS_Microbiota import Simulation, restore_agent
from MAS_Microbiota.Environments.Microbiota.Agents.Bacterium import Bacterium
from repast4py.space import DiscretePoint as dpt

from MAS_Microbiota.Environments.Microbiota.Agents.Metabolite import Metabolite
from MAS_Microbiota.Environments.Microbiota.Agents.SCFA import SCFAType
from MAS_Microbiota.Environments.Microbiota.Agents.Substrate import Substrate


class Microbiota:

    def __init__(self):
        self.context = Simulation.model.envs['microbiota']['context']
        self.grid = Simulation.model.envs['microbiota']['grid']

    def step(self):
        self.context.synchronize(restore_agent)
        self.action()
        self.context.synchronize(restore_agent)

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

    def perform_action(self, bacteria, neighbours):
        pairs = list(zip(bacteria, neighbours))
        for i, (bacterium, neighbours_list) in enumerate(pairs):
            if bacterium.energy == Energy.MAXIMUM and any(isinstance(neighbour, (SCFAType, Metabolite)) for neighbour in neighbours_list):
                self.fission(bacterium)
            elif Energy.NONE < bacterium.energy < Energy.MAXIMUM and any(isinstance(neighbour, Substrate) for neighbour in neighbours_list):
                self.ferment(bacterium)
            elif bacterium.energy < Energy.MAXIMUM and any(isinstance(neighbour, SCFAType) for neighbour in neighbours_list):
                self.consume(bacterium)
            elif bacterium.energy >= Energy.HIGH and not any(isinstance(neighbour, (SCFAType, Metabolite)) for neighbour in neighbours_list):
                self.bacteriocins(bacterium)


    def remove_dead_bacteria(self):
        dead_bacteria = [agent for agent in self.context.agents() if isinstance(agent, Bacterium) and agent.dead]
        for bacterium in dead_bacteria:
            self.context.remove(bacterium)

    def add_bacteria(self):
        for agent in self.context.agents():
            if isinstance(agent, Bacterium):
                self.fission(agent)

    def fission(self, bacterium):
        point = self.grid.get_random_local_pt(Simulation.model.rng)
        new_bacterium = Bacterium(self.context, Energy.HIGH)
        self.context.agents().add(new_bacterium)

    def ferment(self, bacterium):
        pass

    def consume(self, bacterium):
        pass

    def bacteriocins(self, bacterium):
        pass