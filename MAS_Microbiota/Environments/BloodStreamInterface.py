from repast4py import random
from MAS_Microbiota import Simulation

class BloodStreamInterface():
    def __init__(self, gut_context, brain_context):
        self.gut_context = gut_context
        self.brain_context = brain_context
        self.gut_grid = self.gut_context.get_projection("gut_grid")
        self.brain_grid = self.brain_context.get_projection("brain_grid")
        random.seed = Simulation.params['seed']
        self.rng = random.default_rng

    # Unidirectional channel from gut to brain
    def transfer_from_gut_to_brain(self, agent):
        self.brain_context.add(agent)
        pt = self.brain_grid.get_random_local_pt(self.rng)
        self.brain_grid.move(agent, pt)
        agent.context = 'brain'
        agent.toRemove = False
        agent.toMove = False
        self.gut_context.remove(agent)