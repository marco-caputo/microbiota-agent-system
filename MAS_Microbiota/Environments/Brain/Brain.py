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
    def initial_agents():
        return [
                ('neuron_healthy.count', Neuron, NeuronState.HEALTHY),
                ('neuron_damaged.count', Neuron, NeuronState.DAMAGED),
                ('neuron_dead.count', Neuron, NeuronState.DEAD),
                ('resting_microglia.count', Microglia, MicrogliaState.RESTING),
                ('active_microglia.count', Microglia, MicrogliaState.ACTIVE),
                ('alpha_syn_cleaved_brain.count', CleavedProtein, ProteinName.ALPHA_SYN),
                ('tau_cleaved_brain.count', CleavedProtein, ProteinName.TAU),
                ('alpha_syn_oligomer_brain.count', Oligomer, ProteinName.ALPHA_SYN),
                ('tau_oligomer_brain.count', Oligomer, ProteinName.TAU),
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
        all_true_cleaved_aggregates = []
        neurons = []
        active_microglias = 0
        damaged_neurons = 0

        for agent in self.context.agents():
            if isinstance(agent, Oligomer) and agent.toRemove:
                oligomer_to_remove.append(agent)
            elif isinstance(agent, Microglia) and agent.state == MicrogliaState.ACTIVE:
                active_microglias += 1
            elif isinstance(agent, Neuron):
                if agent.state == NeuronState.DAMAGED: damaged_neurons += 1
                neurons.append(agent)
            elif isinstance(agent, CleavedProtein) and agent.toAggregate:
                all_true_cleaved_aggregates.append(agent)
                agent.toRemove = True

        self.produce_neurotransmitters(neurons)
        self.add_cytokines(active_microglias)
        self.brain_add_cleaved_protein(damaged_neurons)
        self.remove_oligomers(removed_ids, oligomer_to_remove)
        self.context.synchronize(restore_agent)
        self.aggreagate_cleaved_proteins(removed_ids, all_true_cleaved_aggregates)
        self.context.synchronize(restore_agent)
        self.remove_agents(removed_ids)

    def produce_neurotransmitters(self, neurons):
        produced_neurotrans = {n_type: 0 for n_type in list(NeurotransmitterType)}

        for neuron in neurons:
            for n_type in NeurotransmitterType:
                if neuron.neurotrans_availability[n_type] > 0:
                    produced_neurotrans[n_type] += neuron.neurotrans_rate[n_type]

        for n_type in produced_neurotrans:
            self.context.add(Neurotransmitter(
                Simulation.model.new_id(),
                Simulation.model.rank,
                n_type,
                self.grid.get_random_local_pt(Simulation.model.rng),
                self.NAME
            ))


    def add_cytokines(self, active_microglias: int):
        """
        Adds a cytokine agent to the brain context for each active microglia.
        :param active_microglias: Number of active microglias.
        """
        for _ in range(active_microglias):
            pt = self.grid.get_random_local_pt(Simulation.model.rng)
            cytokine = Cytokine(Simulation.model.new_id(), Simulation.model.rank, pt, 'brain')
            self.context.add(cytokine)
            Simulation.model.move(cytokine, cytokine.pt, 'brain')


    # Function to add a cleaved protein agent to the brain context for each damaged neuron
    def brain_add_cleaved_protein(self, damaged_neurons: int):
        """
        Adds a cleaved protein agent to the brain context for each damaged neuron.
        :param damaged_neurons: Number of damaged neurons.
        """
        for _ in range(damaged_neurons):
            cleaved_protein_name = Simulation.model.rng.choice(list(ProteinName))
            pt = self.grid.get_random_local_pt(Simulation.model.rng)
            cleaved_protein = CleavedProtein(Simulation.model.new_id(), Simulation.model.rank,
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