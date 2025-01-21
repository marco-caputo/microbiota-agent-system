import numpy as np

from .Agents import *
from MAS_Microbiota import Simulation
from MAS_Microbiota.AgentRestorer import restore_agent
from .. import GridEnvironment


class Gut(GridEnvironment):
    NAME = 'gut'

    def __init__(self, context, grid):
        super().__init__(context, grid)

    @staticmethod
    def agent_types():
        return [
            ('aep_enzyme.count', AEP, None),
            ('tau_proteins.count', Protein, Simulation.params["protein_name"]["tau"]),
            ('alpha_syn_proteins.count', Protein, Simulation.params["protein_name"]["alpha_syn"]),
            ('external_input.count', ExternalInput, None),
            ('treatment_input.count', Treatment, None),
            ('alpha_syn_oligomers_gut.count', Oligomer, Simulation.params["protein_name"]["alpha_syn"]),
            ('tau_oligomers_gut.count', Oligomer, Simulation.params["protein_name"]["tau"]),
        ]

    # Function to check if the microbiota is dysbiotic and adjust the barrier impermeability
    def microbiota_dysbiosis_step(self):
        if (Simulation.model.microbiota_good_bacteria_class - Simulation.model.microbiota_pathogenic_bacteria_class
                <= Simulation.model.microbiota_diversity_threshold):
            value_decreased = int((Simulation.params["barrier_impermeability"] * np.random.randint(0, 6)) / 100)
            if Simulation.model.barrier_impermeability - value_decreased <= 0:
                Simulation.model.barrier_impermeability = 0
            else:
                Simulation.model.barrier_impermeability = Simulation.model.barrier_impermeability - value_decreased
            number_of_aep_to_hyperactivate = value_decreased
            cont = 0
            for agent in self.context.agents(agent_type=AEP.TYPE):
                if agent.state == Simulation.params["aep_state"]["active"] and cont < number_of_aep_to_hyperactivate:
                    agent.state = Simulation.params["aep_state"]["hyperactive"]
                    cont += 1
                elif cont == number_of_aep_to_hyperactivate:
                    break
        else:
            if Simulation.model.barrier_impermeability < Simulation.params["barrier_impermeability"]:
                value_increased = int((Simulation.params["barrier_impermeability"] * np.random.randint(0, 4)) / 100)
                if (Simulation.model.barrier_impermeability + value_increased) <= Simulation.params["barrier_impermeability"]:
                    Simulation.model.barrier_impermeability = Simulation.model.barrier_impermeability + value_increased


    def agents_to_remove(self):
        return (Oligomer, CleavedProtein, Protein)


    def step(self):
        removed_ids = set()
        self.context.synchronize(restore_agent)
        self.remove_agents(removed_ids)
        self.context.synchronize(restore_agent)
        self.make_agents_steps()

        # Collect data and perform operations based on agent states
        oligomers_to_move = []
        proteins_to_remove = []
        all_true_cleaved_aggregates = []

        for agent in self.context.agents():
            if type(agent) == Protein and agent.toCleave == True:
                proteins_to_remove.append(agent)
                agent.toRemove = True
            elif type(agent) == CleavedProtein and agent.toAggregate == True:
                all_true_cleaved_aggregates.append(agent)
                agent.toRemove = True
            elif type(agent) == Oligomer and agent.toMove == True:
                oligomers_to_move.append(agent)
                agent.toRemove = True

        self.move_oligomers_to_brain(oligomers_to_move)
        self.remove_proteins_and_add_cleaved_proteins(removed_ids, proteins_to_remove)
        self.context.synchronize(restore_agent)
        self.aggreagate_cleaved_proteins(removed_ids, all_true_cleaved_aggregates)
        self.context.synchronize(restore_agent)
        self.remove_agents(removed_ids)



    def move_oligomers_to_brain(self, oligomers_to_move):
        for agent in oligomers_to_move:
            Simulation.model.gutBrainInterface.transfer_from_gut_to_brain(agent)


    def remove_proteins_and_add_cleaved_proteins(self, removed_ids, protein_to_remove):
        for agent in protein_to_remove:
            if agent.uid in removed_ids:
                continue
            protein_name = agent.name
            Simulation.model.remove_agent(agent)
            removed_ids.add(agent.uid)
            Simulation.model.gut_add_cleaved_protein(protein_name)
            Simulation.model.gut_add_cleaved_protein(protein_name)


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
                Simulation.model.add_oligomer_protein(agent.name, 'gut')
                Simulation.model.remove_agent(agent)
                removed_ids.add(agent.uid)
