from typing import List

import numpy as np
import pygame
from mpi4py import MPI
from repast4py import context as ctx
from repast4py import space, schedule, logging, random
from repast4py.space import DiscretePoint as dpt

from MAS_Microbiota.GUI import GUI
from MAS_Microbiota.Utils import *
#from MAS_Microbiota.Log import Log #TODO: temporaneo
from MAS_Microbiota.Environments.Gut.Gut import Gut
from MAS_Microbiota.Environments.Brain.Brain import Brain
from MAS_Microbiota.Environments.Microbiota.Microbiota import Microbiota
from MAS_Microbiota.Environments.GutBrainInterface import GutBrainInterface
from MAS_Microbiota.Environments.Microbiota.Agents import *
from MAS_Microbiota.Environments.Gut.Agents import *
from MAS_Microbiota.Environments.Brain.Agents import *
from MAS_Microbiota.AgentRestorer import restore_agent


class Model():

    # Initialize the model
    def __init__(self, comm: MPI.Intracomm):
        self.comm = comm
        self.rank = comm.Get_rank() #Process rank id ranging from 0 to world_size-1
        self.world_size = self.comm.Get_size()  # Number of processes participating in the simulation when using the MPI

        self.init_environments(comm)
        self.init_gui()
        #self.init_log() #TODO: temporaneo
        self.init_schedule(comm)
        self.init_rng()

        self.update_microbiota_params()
        self.init_brain_params()
        self.init_gut_brain_interface_params()

        # Initialize the agents
        self.added_agents_id = 0
        self.distribute_all_agents(Gut.initial_agents(), Gut.NAME)
        self.distribute_all_agents(Brain.initial_agents(), Brain.NAME)
        self.distribute_all_agents(Microbiota.initial_agents(), Microbiota.NAME) #TODO: magari distribuiamo abtteri della stessa famiglia vicini

        # Synchronize the contexts
        for _, env in self.envs.items(): env.context.synchronize(restore_agent)

    def init_environments(self, comm: MPI.Intracomm):
        """
        Initializes the environments for the model, setting up their shared contexts and grids.
        :param comm: MPI communicator
        """
        # Initialization of environment dictionary
        self.envs = dict()

        # Create box grid for the environments
        box = space.BoundingBox(0, Simulation.params['world.width'] - 1, 0, Simulation.params['world.height'] - 1, 0, 0)

        # Create shared contexts and grid for the environments
        for Env in [Microbiota, Gut, Brain]:
            context = ctx.SharedContext(comm)
            grid = self.init_grid(Env.NAME+'_grid', box, context)
            self.envs[Env.NAME] = Env(context, grid)

        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)
        self.gutBrainInterface = GutBrainInterface(self.envs)


    def init_grid(self, name, box, context):
        """
        Initializes a shared grid for the model.
        :param name: The name of the grid
        :param box: The bounding box of the grid
        :param context: The context to which the grid belongs
        :return: The initialized grid
        """
        grid = space.SharedGrid(name=name, bounds=box, borders=space.BorderType.Sticky,
                                occupancy=space.OccupancyType.Multiple, buffer_size=1, comm=self.comm)
        context.add_projection(grid)
        return grid

    def init_gui(self):
        """
        Initializes the pygame GUI and its objects for the model.
        """
        pygame.init()
        self.screen = GUI(width=1600, height=800, envs=self.envs)
        pygame.display.set_caption("Gut-Brain Axis Model")
        self.screen.update()


    def init_schedule(self, comm: MPI.Intracomm):
        """
        Initializes the schedule for the model.
        The schedule represents the order in which the model's events are executed in each tick of the simulation.
        """
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.envs[Microbiota.NAME].step)
        self.runner.schedule_repeating_event(1, 1, self.update_microbiota_params)
        self.runner.schedule_repeating_event(1, 1, self.envs[Gut.NAME].step)
        self.runner.schedule_repeating_event(1, 2, self.envs[Gut.NAME].microbiota_dysbiosis_step)
        self.runner.schedule_repeating_event(1, 6, self.teleport_resources_step)
        self.runner.schedule_repeating_event(1, 1, self.envs[Brain.NAME].step, priority_type=0)
        self.runner.schedule_repeating_event(1, 1, self.screen.pygame_update, priority_type=1)
        #self.runner.schedule_repeating_event(1, 1, self.counts.log_counts, priority_type=1) #TODO: temporaneo
        self.runner.schedule_stop(Simulation.params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

    def init_rng(self):
        """
        Initializes the random number generators for the model.
        """
        random.seed = Simulation.params['seed']
        self.rng: np.random.Generator = random.default_rng

    def init_log(self):
        """
        Initializes the log for the model.
        """
        self.counts = Log()
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, self.comm, Simulation.params['log_file'], buffer_size=1)

    def update_microbiota_params(self):
        """
        Updates the microbiota parameters for the model.
        """
        self.microbiota_good_bacteria_count = self.envs[Microbiota.NAME].good_bacteria_count
        self.microbiota_pathogenic_bacteria_count = self.envs[Microbiota.NAME].pathogenic_bacteria_count

    def init_gut_brain_interface_params(self):
        self.epithelial_barrier_impermeability = \
            Simulation.params["epithelial_barrier"]["initial_impermeability"]
        self.epithelial_barrier_permeability_threshold_stop = \
            Simulation.params["epithelial_barrier"]["permeability_threshold_stop"]
        self.epithelial_barrier_permeability_threshold_start = \
            Simulation.params["epithelial_barrier"]["permeability_threshold_start"]

    def init_brain_params(self):
        """
        Initializes the brain parameters for the model.
        """
        self.pro_cytokine = 0
        self.anti_cytokine = 0
        self.dead_neuron = self.calculate_partitioned_count(Simulation.params['neuron_dead.count'])

    # Function to distribute all agents through the different ranks
    def distribute_all_agents(self, agent_types: List[tuple], env):
        """
        Distributes all agents through the different ranks.
        :param agent_types: List of tuples containing the agent type, count, and initial state key
        :param env: Environment where to distribute the agents
        """
        for agent_type in agent_types:
            total_count = Simulation.params[agent_type[0]]
            pp_count = self.calculate_partitioned_count(total_count)
            self.create_agents(agent_type[1], pp_count, agent_type[2], env)

    # Function to create agents in the different ranks based on the total count
    def create_agents(self, agent_class, pp_count, state, env_name):
        for j in range(pp_count):
            pt = self.envs[env_name].grid.get_random_local_pt(self.rng)
            if agent_class in [Neuron, Microglia, CleavedProtein, Oligomer, Protein, SCFA, Substrate, ExternalInput, Treatment]:
                agent = agent_class(self.added_agents_id + j, self.rank, state, pt, env_name)
            else:
                # For agents without special state
                agent = agent_class(self.added_agents_id + j, self.rank, pt, env_name)
            self.envs[env_name].context.add(agent)
            self.move(agent, pt, agent.context)
        self.added_agents_id += pp_count

    # Function to get the total count of agents to create in that rank
    def calculate_partitioned_count(self, total_count):
        pp_count = int(total_count / self.world_size)
        if self.rank < total_count % self.world_size:
            pp_count += 1
        return pp_count


    # Function to remove an agent from the context and the grid
    def remove_agent(self, agent):
        self.envs[agent.context].context.remove(agent)


    # Function to add a cleaved protein agent to the gut context
    def gut_add_cleaved_protein(self, cleaved_protein_name):
        self.added_agents_id += 1
        pt = self.envs[Gut.NAME].grid.get_random_local_pt(self.rng)
        cleaved_protein = CleavedProtein(self.added_agents_id, self.rank, cleaved_protein_name, pt, 'gut')
        self.envs[Gut.NAME].context.add(cleaved_protein)
        self.move(cleaved_protein, cleaved_protein.pt, 'gut')


    # Function to move the cleaved protein agents
    def teleport_resources_step(self):
        self.teleport_cleaved_protein_step()
        #self.envs[Microbiota.NAME].teleport_resources_step()

    def teleport_cleaved_protein_step(self):
        for env_name in [Gut.NAME, Brain.NAME]:
            for agent in Simulation.model.envs[env_name].context.agents():
                if type(agent) == CleavedProtein and not agent.alreadyAggregate:
                    pt = Simulation.model.envs[env_name].grid.get_random_local_pt(Simulation.model.rng)
                    Simulation.model.move(agent, pt, agent.context)


    # Function to add an oligomer protein agent to the brain or gut context
    def add_oligomer_protein(self, oligomer_name, env_name):
        self.added_agents_id += 1
        pt = self.envs[env_name].grid.get_random_local_pt(self.rng)
        oligomer_protein = Oligomer(self.added_agents_id, self.rank, oligomer_name, pt, env_name)
        self.envs[env_name].context.add(oligomer_protein)
        self.move(oligomer_protein, oligomer_protein.pt, env_name)


    # Function to move an agent to a new location
    def move(self, agent, pt: dpt, env_name):
        self.envs[env_name].grid.move(agent, pt)
        agent.pt = pt

    def new_id(self):
        """
        Generates a new unique id for an agent.
        :return: The new unique id
        """
        self.added_agents_id += 1
        return self.added_agents_id

    # Function to close the data set and quit Pygame
    def at_end(self):
        self.data_set.close()
        pygame.quit()

    # Function to start the simulation
    def start(self):
        self.runner.execute()

    # Static function to run the simulation
    @staticmethod
    def run():
        model = Model(MPI.COMM_WORLD)
        Simulation.set_model(model)
        model.start()
