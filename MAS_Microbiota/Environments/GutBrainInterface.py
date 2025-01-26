from repast4py import random
from MAS_Microbiota import Simulation
from MAS_Microbiota.Environments import GridAgent
from MAS_Microbiota.Environments.Brain.Agents import Precursor
from MAS_Microbiota.Environments.Brain.Brain import Brain
from MAS_Microbiota.Environments.Gut.Agents import Oligomer
from MAS_Microbiota.Environments.Microbiota.Agents import SCFA


class GutBrainInterface:
    def __init__(self, envs: dict):
        self.envs = envs
        random.seed = Simulation.params['seed']
        self.rng = random.default_rng

    # Unidirectional channel from gut to brain
    def transfer_to_bloodstream(self, agent: GridAgent):
        if isinstance(agent, Oligomer):
            self.transfer_to_brain(agent)
        elif isinstance(agent, SCFA):
            #TODO: change BBB impermeability #TODO: clean simulation parameters
            pass
        elif isinstance(agent, Precursor):
            #TODO: put into the environment based on BBB impermeability
            pass


    def transfer_to_brain(self, agent: GridAgent):
        original_env_name = agent.context
        self.envs[Brain.NAME].context.add(agent)
        pt = self.envs[Brain.NAME].grid.get_random_local_pt(self.rng)
        Simulation.model.move(agent, pt, Brain.NAME)
        agent.context = Brain.NAME
        agent.toRemove = False
        agent.toMove = False
        self.envs[original_env_name].remove(agent)