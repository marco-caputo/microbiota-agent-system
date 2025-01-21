import numpy as np

from .Agents import *
from MAS_Microbiota.Environments.Gut.Agents import *
from MAS_Microbiota.AgentRestorer import restore_agent
from MAS_Microbiota import Simulation
from .. import GridEnvironment


class Brain(GridEnvironment):
    NAME = 'brain'

    def __init__(self, context, grid):
        super().__init__(context, grid)

    @staticmethod
    def agent_types():
        return [
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

    def agents_to_remove(self):
        return (Oligomer, CleavedProtein, Neuron)


    # Brain steps
    def step(self):
        removed_ids = set()
        self.context.synchronize(restore_agent)
        self.remove_agents(removed_ids)
        self.context.synchronize(restore_agent)
        self.make_agents_steps()

        # Collect data and perform operations based on agent states
        oligomer_to_remove = []
        active_microglias = 0
        damaged_neurons = 0
        all_true_cleaved_aggregates = []

        for agent in self.context.agents():
            if isinstance(agent, Oligomer) and agent.toRemove:
                oligomer_to_remove.append(agent)
            elif isinstance(agent, Microglia) and agent.state == Simulation.params["microglia_state"]["active"]:
                active_microglias += 1
            elif isinstance(agent, Neuron) and agent.state == Simulation.params["neuron_state"]["damaged"]:
                damaged_neurons += 1
            elif isinstance(agent, CleavedProtein) and agent.toAggregate:
                all_true_cleaved_aggregates.append(agent)
                agent.toRemove = True

        self.add_cytokines(active_microglias)
        self.brain_add_cleaved_protein(damaged_neurons)
        self.remove_oligomers(removed_ids, oligomer_to_remove)
        self.context.synchronize(restore_agent)
        self.aggreagate_cleaved_proteins(removed_ids, all_true_cleaved_aggregates)
        self.context.synchronize(restore_agent)
        self.remove_agents(removed_ids)


    # Function to add a cytokine agent to the brain context
    def add_cytokines(self, active_microglias: int):
        for _ in range(active_microglias):
            Simulation.model.added_agents_id += 1
            pt = self.grid.get_random_local_pt(Simulation.model.rng)
            cytokine = Cytokine(Simulation.model.added_agents_id, Simulation.model.rank, pt, 'brain')
            self.context.add(cytokine)
            Simulation.model.move(cytokine, cytokine.pt, 'brain')


    # Function to add a cleaved protein agent to the brain context for each damaged neuron
    def brain_add_cleaved_protein(self, damaged_neurons):
        for _ in range(damaged_neurons):
            Simulation.model.added_agents_id += 1
            possible_types = [Simulation.params["protein_name"]["alpha_syn"], Simulation.params["protein_name"]["tau"]]
            random_index = np.random.randint(0, len(possible_types))
            cleaved_protein_name = possible_types[random_index]
            pt = self.grid.get_random_local_pt(Simulation.model.rng)
            cleaved_protein = CleavedProtein(Simulation.model.added_agents_id, Simulation.model.rank,
                                             cleaved_protein_name, pt, 'brain')
            self.context.add(cleaved_protein)
            Simulation.model.move(cleaved_protein, cleaved_protein.pt, cleaved_protein.context)


    def remove_oligomers(self, removed_ids, oligomer_to_remove):
        for oligomer in oligomer_to_remove:
            if self.context.agent(oligomer.uid) is not None:
                Simulation.model.remove_agent(oligomer)
                removed_ids.add(oligomer.uid)


    def aggreagate_cleaved_proteins(self, removed_ids, all_true_cleaved_aggregates):
        for agent in all_true_cleaved_aggregates:
            if agent.uid in removed_ids:
                continue
            if agent.toAggregate and agent.is_valid():
                cont = 0
                _, agent_nghs_cleaved, _ = agent.check_and_get_nghs()
                for x in agent_nghs_cleaved:
                    if x.alreadyAggregate and x.uid != agent.uid:
                        if cont < 3:
                            if self.context.agent(x.uid) is not None:
                                Simulation.model.remove_agent(x)
                                removed_ids.add(x.uid)
                            cont += 1
                        else:
                            x.alreadyAggregate = False
                            x.toAggregate = False
                            cont += 1
                Simulation.model.add_oligomer_protein(agent.name, agent.context)
                Simulation.model.remove_agent(agent)
                removed_ids.add(agent.uid)