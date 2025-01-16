from typing import List

import pygame
from mpi4py import MPI
from repast4py import context as ctx
from repast4py import space, schedule, logging, random
from repast4py.space import DiscretePoint as dpt

from MAS_Microbiota.GUI import GUI
from MAS_Microbiota.Utils import *
from MAS_Microbiota.Log import Log
from MAS_Microbiota.Environments.GutBrainInterface import GutBrainInterface
from MAS_Microbiota.Environments.Gut.Agents import *
from MAS_Microbiota.Environments.Brain.Agents import *
from MAS_Microbiota.AgentRestorer import restore_agent
from MAS_Microbiota.Environments.Gut import Gut
from MAS_Microbiota.Environments.Brain import Brain


class Model():

    # Initialize the model
    def __init__(self, comm: MPI.Intracomm):
        self.comm = comm
        self.rank = comm.Get_rank() #Process rank id ranging from 0 to world_size-1
        self.world_size = self.comm.Get_size()  # Number of processes participating in the simulation when using the MPI

        self.init_environments(comm)
        self.init_gui()
        self.init_log()
        self.init_schedule(comm)
        self.init_rng()

        self.init_microbiota_params() #TODO: Agire qui: I parametri su batteri patogeni e buoni devono coincidere con il conteggio degli agenti. Inoltre si potrebbe anche pensare di non usare un threshold dei batteri ma piuttosto dell'infiammazione.
        self.init_brain_params()

        # Initialize the agents
        self.added_agents_id = 0
        self.distribute_all_agents(Gut.agent_types(), 'gut')
        self.distribute_all_agents(Brain.agent_types(), 'brain')

        # Synchronize the contexts
        for env in self.envs: self.envs[env]["context"].synchronize(restore_agent)

    def init_environments(self, comm: MPI.Intracomm):
        """
        Initializes the environments for the model, setting up their shared contexts and grids.
        :param comm: MPI communicator
        """
        # Initialization of environment dictionary for contexts and grids
        self.envs = {"microbiota": dict(), "gut": dict(), "brain": dict()}

        # Create box grid for the environments
        box = space.BoundingBox(0, Simulation.params['world.width'] - 1, 0, Simulation.params['world.height'] - 1, 0, 0)

        # Create shared contexts and grid for the environments
        for env in self.envs:
            self.envs[env]["context"] = ctx.SharedContext(comm)
            self.envs[env]["grid"] = self.init_grid(env+'_grid', box, self.envs[env]["context"])

        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)
        self.gutBrainInterface = GutBrainInterface(self.envs["gut"]["context"], self.envs["brain"]["context"])


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
        self.runner.schedule_repeating_event(1, 1, Gut.step)
        self.runner.schedule_repeating_event(1, 2, Gut.microbiota_dysbiosis_step)
        self.runner.schedule_repeating_event(1, 5, Gut.move_cleaved_protein_step)
        self.runner.schedule_repeating_event(1, 1, Brain.step, priority_type=0)
        self.runner.schedule_repeating_event(1, 1, self.screen.pygame_update, priority_type=1)
        self.runner.schedule_repeating_event(1, 1, self.counts.log_counts, priority_type=1)
        self.runner.schedule_stop(Simulation.params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

    def init_rng(self):
        """
        Initializes the random number generator for the model.
        """
        random.seed = Simulation.params['seed']
        self.rng = random.default_rng

    def init_log(self):
        """
        Initializes the log for the model.
        """
        self.counts = Log()
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, self.comm, Simulation.params['log_file'], buffer_size=1)

    def init_microbiota_params(self):
        """
        Initializes the microbiota parameters for the model.
        """
        self.microbiota_good_bacteria_class = Simulation.params["microbiota_good_bacteria_class"]
        self.microbiota_pathogenic_bacteria_class = Simulation.params["microbiota_pathogenic_bacteria_class"]
        self.microbiota_diversity_threshold = Simulation.params["microbiota_diversity_threshold"]
        self.barrier_impermeability = Simulation.params["barrier_impermeability"]
        self.barrier_permeability_threshold_stop = Simulation.params["barrier_permeability_threshold_stop"]
        self.barrier_permeability_threshold_start = Simulation.params["barrier_permeability_threshold_start"]

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
    def create_agents(self, agent_class, pp_count, state_key, env):
        for j in range(pp_count):
            pt = self.envs[env]['grid'].get_random_local_pt(self.rng)
            if agent_class in [Neuron, Microglia]:
                agent = agent_class(self.added_agents_id + j, self.rank,
                                    Simulation.params[f"{agent_class.__name__.lower()}_state"][state_key], pt, env)
            elif agent_class in [CleavedProtein, Oligomer, Protein]:
                agent = agent_class(self.added_agents_id + j, self.rank, Simulation.params["protein_name"][state_key], pt, env)
            else:
                # For agents without special state keys
                agent = agent_class(self.added_agents_id + j, self.rank, pt, env)
            self.envs[env]['context'].add(agent)
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
        self.envs[agent.context]['context'].remove(agent)


    # Function to add a cleaved protein agent to the gut context
    def gut_add_cleaved_protein(self, cleaved_protein_name):
        self.added_agents_id += 1
        pt = self.envs['gut']['grid'].get_random_local_pt(self.rng)
        cleaved_protein = CleavedProtein(self.added_agents_id, self.rank, cleaved_protein_name, pt, 'gut')
        self.envs['gut']['context'].add(cleaved_protein)
        self.move(cleaved_protein, cleaved_protein.pt, 'gut')


    # Function to add an oligomer protein agent to the brain or gut context
    def add_oligomer_protein(self, oligomer_name, env):
        self.added_agents_id += 1
        pt = self.envs[env]['grid'].get_random_local_pt(self.rng)
        oligomer_protein = Oligomer(self.added_agents_id, self.rank, oligomer_name, pt, env)
        self.envs[env]['context'].add(oligomer_protein)
        self.move(oligomer_protein, oligomer_protein.pt, env)


    # Function to move an agent to a new location
    def move(self, agent, pt: dpt, env):
        self.envs[env]['grid'].move(agent, pt)
        agent.pt = pt


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
