from Bacterium import Energy
from MAS_Microbiota import Simulation, restore_agent
from MAS_Microbiota.Environments import GridAgent, GridEnvironment
from MAS_Microbiota.Environments.Microbiota.Agents.Bacterium import Bacterium, State
from MAS_Microbiota.Environments.Microbiota.Agents.Metabolite import Metabolite
from MAS_Microbiota.Environments.Microbiota.Agents.SCFA import SCFAType
from MAS_Microbiota.Environments.Microbiota.Agents.Substrate import Substrate


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
            if bacterium.state == State.FISSION:
                self.fission(bacterium, neighbours)
            elif bacterium.state == State.FERMENT:
                self.ferment(bacterium, neighbours)
            elif bacterium.state == State.CONSUME:
                self.consume(bacterium, neighbours)
            elif bacterium.state == State.BACTERIOCINS:
                self.bacteriocins(bacterium, neighbours)

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

    def perform_action(self, bacteria, neighbours):
        pairs = list(zip(bacteria, neighbours))
        for i, (bacterium, neighbours_list) in enumerate(pairs):
            if bacterium.energy == Energy.MAXIMUM and any(isinstance(neighbour, (SCFAType, Metabolite)) for neighbour in neighbours_list):
                self.fission(bacterium, neighbours_list)
            elif Energy.NONE < bacterium.energy < Energy.MAXIMUM and any(isinstance(neighbour, Substrate) for neighbour in neighbours_list):
                self.ferment(bacterium)
            elif bacterium.energy < Energy.MAXIMUM and any(isinstance(neighbour, SCFAType) for neighbour in neighbours_list):
                self.consume(bacterium)
            elif bacterium.energy >= Energy.HIGH and not any(isinstance(neighbour, (SCFAType, Metabolite)) for neighbour in neighbours_list):
                self.bacteriocins(bacterium)


    def remove_dead_bacteria(self):
        dead_bacteria = [agent for agent in self.context.agents() if isinstance(agent, Bacterium) and agent.toRemove]
        for bacterium in dead_bacteria:
            self.context.remove(bacterium)

    def fission(self, bacterium, neighbours):
        point = neighbours.get_random_local_pt(Simulation.model.rng)
        new_bacterium = Bacterium(self.context, Energy.HIGH, point, State.IDLE)
        self.context.agents().add(new_bacterium)

    def ferment(self, bacterium, neighbours):
        point = neighbours.get_random_local_pt(Simulation.model.rng)
        scfa = SCFAType(self.context, Energy.HIGH, point, State.IDLE)
        for neighbour in (neighbour for neighbour in neighbours if isinstance(neighbour, Substrate)):
            neighbour.toRemove = True
            break
        self.context.agents().add(scfa)

    def consume(self, bacterium, neighbours):
        for neighbour in (neighbour for neighbour in neighbours if isinstance(neighbour, SCFAType)):
            neighbour.toRemove = True

    def bacteriocins(self, bacterium, neighbours):
        for neighbour in (neighbour for neighbour in neighbours if isinstance(neighbour, Bacterium)):
            neighbour.toRemove = True