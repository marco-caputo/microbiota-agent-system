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
from MAS_Microbiota.Environments.Gut.Gut import gut_step, microbiota_dysbiosis_step, move_cleaved_protein_step
from MAS_Microbiota.Environments.Brain.Brain import brain_step


class Model():

    # Initialize the model
    def __init__(self, comm: MPI.Intracomm):
        self.comm = comm
        self.rank = comm.Get_rank()
        # Create shared contexts for the brain and the gut
        self.gut_context = ctx.SharedContext(comm)
        self.brain_context = ctx.SharedContext(comm)
        # Create shared grids for the brain and the gut
        box = space.BoundingBox(0, Simulation.params['world.width'] - 1, 0, Simulation.params['world.height'] - 1, 0, 0)
        self.gut_grid = self.init_grid('gut_grid', box, self.gut_context)
        self.brain_grid = self.init_grid('brain_grid', box, self.brain_context)

        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)
        self.gutBrainInterface = GutBrainInterface(self.gut_context, self.brain_context)
        # Initialize the schedule runner
        self.runner = schedule.init_schedule_runner(comm)
        self.init_schedule()
        # Set the seed for the random number generator
        random.seed = Simulation.params['seed']
        self.rng = random.default_rng
        # Initialize the log
        self.counts = Log()
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, self.comm, Simulation.params['log_file'], buffer_size=1)
        # Initialize the model parameters
        self.init_microbiota_params()
        self.world_size = self.comm.Get_size()
        self.added_agents_id = 0
        self.pro_cytokine = 0
        self.anti_cytokine = 0
        self.dead_neuron = self.calculate_partitioned_count(Simulation.params['neuron_dead.count'])
        # Initialize the agents
        agent_types_gut = [
            ('aep_enzyme.count', AEP, None),
            ('tau_proteins.count', Protein, Simulation.params["protein_name"]["tau"]),
            ('alpha_syn_proteins.count', Protein, Simulation.params["protein_name"]["alpha_syn"]),
            ('external_input.count', ExternalInput, None),
            ('treatment_input.count', Treatment, None),
            ('alpha_syn_oligomers_gut.count', Oligomer, Simulation.params["protein_name"]["alpha_syn"]),
            ('tau_oligomers_gut.count', Oligomer, Simulation.params["protein_name"]["tau"]),
        ]
        agent_types_brain = [
            ('neuron_healthy.count', Neuron, 'healthy'),
            ('neuron_damaged.count', Neuron, 'damaged'),
            ('neuron_dead.count', Neuron, 'dead'),
            ('resting_microglia.count', Microglia, 'resting'),
            ('active_microglia.count', Microglia, 'active'),
            ('alpha_syn_cleaved_brain.count', CleavedProtein, Simulation.params["protein_name"]["alpha_syn"]),
            ('tau_cleaved_brain.count', CleavedProtein, Simulation.params["protein_name"]["tau"]),
            ('alpha_syn_oligomer_brain.count', Oligomer, Simulation.params["protein_name"]["alpha_syn"]),
            ('tau_oligomer_brain.count', Oligomer, Simulation.params["protein_name"]["tau"]),
            ('cytokine.count', Cytokine, None)
        ]
        self.distribute_all_agents(agent_types_gut, self.gut_context, self.gut_grid, 'gut')
        self.distribute_all_agents(agent_types_brain, self.brain_context, self.brain_grid, 'brain')
        # Synchronize the contexts
        self.gut_context.synchronize(restore_agent)
        self.brain_context.synchronize(restore_agent)
        # Initialize Pygame and gui object
        pygame.init()
        self.screen = GUI(width=1600, height=800, gut_context=self.gut_context, brain_context=self.brain_context)
        pygame.display.set_caption("Gut-Brain Axis Model")
        self.screen.update(gut_context=self.gut_context, brain_context=self.brain_context)

    # Function to initialize the shared grid
    def init_grid(self, name, box, context):
        grid = space.SharedGrid(name=name, bounds=box, borders=space.BorderType.Sticky,
                                occupancy=space.OccupancyType.Multiple, buffer_size=1, comm=self.comm)
        context.add_projection(grid)
        return grid

    # Function to initialize the schedule
    def init_schedule(self):
        self.runner.schedule_repeating_event(1, 1, gut_step)
        self.runner.schedule_repeating_event(1, 2, microbiota_dysbiosis_step)
        self.runner.schedule_repeating_event(1, 5, move_cleaved_protein_step)
        self.runner.schedule_repeating_event(1, 1, brain_step, priority_type=0)
        self.runner.schedule_repeating_event(1, 1, self.pygame_update, priority_type=1)
        self.runner.schedule_repeating_event(1, 1, self.log_counts, priority_type=1)
        self.runner.schedule_stop(Simulation.params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

    # Function to initialize the microbiota parameters
    def init_microbiota_params(self):
        self.microbiota_good_bacteria_class = Simulation.params["microbiota_good_bacteria_class"]
        self.microbiota_pathogenic_bacteria_class = Simulation.params["microbiota_pathogenic_bacteria_class"]
        self.microbiota_diversity_threshold = Simulation.params["microbiota_diversity_threshold"]
        self.barrier_impermeability = Simulation.params["barrier_impermeability"]
        self.barrier_permeability_threshold_stop = Simulation.params["barrier_permeability_threshold_stop"]
        self.barrier_permeability_threshold_start = Simulation.params["barrier_permeability_threshold_start"]

    # Function to distribute all agents through the different ranks
    def distribute_all_agents(self, agent_types, context, grid, region):
        for agent_type in agent_types:
            total_count = Simulation.params[agent_type[0]]
            pp_count = self.calculate_partitioned_count(total_count)
            self.create_agents(agent_type[1], pp_count, agent_type[2], context, grid, region)

    # Function to create agents in the different ranks based on the total count
    def create_agents(self, agent_class, pp_count, state_key, context, grid, region):
        for j in range(pp_count):
            pt = grid.get_random_local_pt(self.rng)
            if agent_class in [Neuron, Microglia]:
                agent = agent_class(self.added_agents_id + j, self.rank,
                                    Simulation.params[f"{agent_class.__name__.lower()}_state"][state_key], pt, region)
            elif agent_class in [CleavedProtein, Oligomer, Protein]:
                agent = agent_class(self.added_agents_id + j, self.rank, Simulation.params["protein_name"][state_key], pt, region)
            else:
                # For agents without special state keys
                agent = agent_class(self.added_agents_id + j, self.rank, pt, region)
            context.add(agent)
            self.move(agent, pt, agent.context)
        self.added_agents_id += pp_count

    # Function to get the total count of agents to create in that rank
    def calculate_partitioned_count(self, total_count):
        pp_count = int(total_count / self.world_size)
        if self.rank < total_count % self.world_size:
            pp_count += 1
        return pp_count

    # Function to update the interface
    def pygame_update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # If the 'X' button is clicked, stop the simulation
                print("Ending the simulation.")
                self.at_end()
                self.comm.Abort()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.screen.handle_button_click(event.pos)

        while self.screen.paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # If the 'X' button is clicked, stop the simulation
                    print("Ending the simulation.")
                    self.at_end()
                    self.comm.Abort()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.screen.handle_button_click(event.pos)

        # Updates the Pygame GUI based on the current state of the Repast simulation
        self.screen.update(gut_context=self.gut_context, brain_context=self.brain_context)
        pygame.display.flip()


    # Function to remove an agent from the context and the grid
    def remove_agent(self, agent):
        if agent.context == 'brain':
            self.brain_context.remove(agent)
        else:
            self.gut_context.remove(agent)


    # Function to add a cleaved protein agent to the gut context
    def gut_add_cleaved_protein(self, cleaved_protein_name):
        self.added_agents_id += 1
        pt = self.gut_grid.get_random_local_pt(self.rng)
        cleaved_protein = CleavedProtein(self.added_agents_id, self.rank, cleaved_protein_name, pt, 'gut')
        self.gut_context.add(cleaved_protein)
        self.move(cleaved_protein, cleaved_protein.pt, 'gut')

    # Function to add an oligomer protein agent to the brain or gut context
    def add_oligomer_protein(self, oligomer_name, context):
        self.added_agents_id += 1
        if context == 'brain':
            pt = self.brain_grid.get_random_local_pt(self.rng)
            oligomer_protein = Oligomer(self.added_agents_id, self.rank, oligomer_name, pt, 'brain')
            self.brain_context.add(oligomer_protein)
            self.move(oligomer_protein, oligomer_protein.pt, 'brain')
        else:
            pt = self.gut_grid.get_random_local_pt(self.rng)
            oligomer_protein = Oligomer(self.added_agents_id, self.rank, oligomer_name, pt, 'gut')
            self.gut_context.add(oligomer_protein)
            self.move(oligomer_protein, oligomer_protein.pt, 'gut')

            # Function to move an agent to a new location

    def move(self, agent, pt: dpt, context):
        if context == 'brain':
            self.brain_grid.move(agent, pt)
        else:
            self.gut_grid.move(agent, pt)
        agent.pt = pt


    # Function to log the counts of the agents
    def log_counts(self):
        tick = self.runner.schedule.tick

        counts = {
            "aep_active": 0,
            "aep_hyperactive": 0,
            "alpha_protein_gut": 0,
            "tau_protein_gut": 0,
            "alpha_cleaved_gut": 0,
            "tau_cleaved_gut": 0,
            "alpha_oligomer_gut": 0,
            "tau_oligomer_gut": 0,
            "microglia_resting": 0,
            "microglia_active": 0,
            "neuron_healthy": 0,
            "neuron_damaged": 0,
            "alpha_cleaved_brain": 0,
            "tau_cleaved_brain": 0,
            "alpha_oligomer_brain": 0,
            "tau_oligomer_brain": 0
        }

        for agent in self.brain_context.agents():
            if isinstance(agent, Oligomer):
                if agent.name == Simulation.params["protein_name"]["alpha_syn"]:
                    counts["alpha_oligomer_brain"] += 1
                else:
                    counts["tau_oligomer_brain"] += 1
            elif isinstance(agent, CleavedProtein):
                if agent.name == Simulation.params["protein_name"]["alpha_syn"]:
                    counts["alpha_cleaved_brain"] += 1
                else:
                    counts["tau_cleaved_brain"] += 1
            elif isinstance(agent, Neuron):
                if agent.state == Simulation.params["neuron_state"]["healthy"]:
                    counts["neuron_healthy"] += 1
                elif agent.state == Simulation.params["neuron_state"]["damaged"]:
                    counts["neuron_damaged"] += 1
            elif isinstance(agent, Microglia):
                if agent.state == Simulation.params["microglia_state"]["active"]:
                    counts["microglia_active"] += 1
                else:
                    counts["microglia_resting"] += 1

        for agent in self.gut_context.agents():
            if (type(agent) == Oligomer):
                if (agent.name == Simulation.params["protein_name"]["alpha_syn"]):
                    counts["alpha_oligomer_gut"] += 1
                else:
                    counts["tau_oligomer_gut"] += 1
            if (type(agent) == CleavedProtein):
                if (agent.name == Simulation.params["protein_name"]["alpha_syn"]):
                    counts["alpha_cleaved_gut"] += 1
                else:
                    counts["tau_cleaved_gut"] += 1
            elif (type(agent) == Protein):
                if (agent.name == Simulation.params["protein_name"]["alpha_syn"]):
                    counts["alpha_protein_gut"] += 1
                else:
                    counts["tau_protein_gut"] += 1
            elif (type(agent) == AEP):
                if (agent.state == Simulation.params["aep_state"]["active"]):
                    counts["aep_active"] += 1
                else:
                    counts["aep_hyperactive"] += 1

                    # brain
        self.counts.healthy_neuron = counts["neuron_healthy"]
        self.counts.damaged_neuron = counts["neuron_damaged"]
        self.counts.dead_neuron = self.dead_neuron
        self.counts.cytokine_pro_inflammatory = self.pro_cytokine
        self.counts.cytokine_non_inflammatory = self.anti_cytokine
        self.counts.cleaved_alpha_syn_brain = counts["alpha_cleaved_brain"]
        self.counts.cleaved_tau_brain = counts["tau_cleaved_brain"]
        self.counts.alpha_syn_oligomer_brain = counts["alpha_oligomer_brain"]
        self.counts.tau_oligomer_brain = counts["tau_oligomer_brain"]
        self.counts.resting_microglia = counts["microglia_resting"]
        self.counts.active_microglia = counts["microglia_active"]

        # gut
        self.counts.aep_active = counts["aep_active"]
        self.counts.aep_hyperactive = counts["aep_hyperactive"]
        self.counts.alpha_protein_gut = counts["alpha_protein_gut"]
        self.counts.tau_protein_gut = counts["tau_protein_gut"]
        self.counts.alpha_cleaved_gut = counts["alpha_cleaved_gut"]
        self.counts.tau_cleaved_gut = counts["tau_cleaved_gut"]
        self.counts.alpha_oligomer_gut = counts["alpha_oligomer_gut"]
        self.counts.tau_oligomer_gut = counts["tau_oligomer_gut"]
        self.counts.microbiota_good_bacteria_class = self.microbiota_good_bacteria_class
        self.counts.microbiota_pathogenic_bacteria_class = self.microbiota_pathogenic_bacteria_class
        self.counts.barrier_impermeability = self.barrier_impermeability

        self.data_set.log(tick)

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
