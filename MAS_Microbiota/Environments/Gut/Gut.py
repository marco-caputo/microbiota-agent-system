import numpy as np

from .Agents import *
from MAS_Microbiota import Simulation
from MAS_Microbiota.AgentRestorer import restore_agent

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


# Function to move the cleaved protein agents
def move_cleaved_protein_step():
    for agent in Simulation.model.envs['gut']['context'].agents():
        if type(agent) == CleavedProtein:
            if agent.alreadyAggregate == False:
                pt = Simulation.model.envs['gut']['grid'].get_random_local_pt(Simulation.model.rng)
                Simulation.model.move(agent, pt, agent.context)
    for agent in Simulation.model.envs['brain']['context'].agents():
        if type(agent) == CleavedProtein:
            if agent.alreadyAggregate == False:
                pt = Simulation.model.envs['brain']['grid'].get_random_local_pt(Simulation.model.rng)
                Simulation.model.move(agent, pt, agent.context)

# Function to check if the microbiota is dysbiotic and adjust the barrier impermeability
def microbiota_dysbiosis_step():
    if Simulation.model.microbiota_good_bacteria_class - Simulation.model.microbiota_pathogenic_bacteria_class <= Simulation.model.microbiota_diversity_threshold:
        value_decreased = int((Simulation.params["barrier_impermeability"] * np.random.randint(0, 6)) / 100)
        if Simulation.model.barrier_impermeability - value_decreased <= 0:
            Simulation.model.barrier_impermeability = 0
        else:
            Simulation.model.barrier_impermeability = Simulation.model.barrier_impermeability - value_decreased
        number_of_aep_to_hyperactivate = value_decreased
        cont = 0
        for agent in Simulation.model.envs['gut']['context'].agents(agent_type=0):
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

def step():
    Simulation.model.envs['gut']['context'].synchronize(restore_agent)

    def gather_agents_to_remove():
        return [agent for agent in Simulation.model.envs['gut']['context'].agents() if
                isinstance(agent, (Oligomer, CleavedProtein, Protein)) and agent.toRemove]

    remove_agents = gather_agents_to_remove()
    removed_ids = set()
    for agent in remove_agents:
        if Simulation.model.envs['gut']['context'].agent(agent.uid) is not None:
            Simulation.model.remove_agent(agent)
            removed_ids.add(agent.uid)

    Simulation.model.envs['gut']['context'].synchronize(restore_agent)

    for agent in Simulation.model.envs['gut']['context'].agents():
        agent.step()

    protein_to_remove = []
    all_true_cleaved_aggregates = []
    oligomers_to_move = []

    for agent in Simulation.model.envs['gut']['context'].agents():
        if (type(agent) == Protein and agent.toCleave == True):
            protein_to_remove.append(agent)
            agent.toRemove = True
        elif (type(agent) == CleavedProtein and agent.toAggregate == True):
            all_true_cleaved_aggregates.append(agent)
            agent.toRemove = True
        elif (type(agent) == Oligomer and agent.toMove == True):
            oligomers_to_move.append(agent)
            agent.toRemove = True

    for agent in oligomers_to_move:
        Simulation.model.gutBrainInterface.transfer_from_gut_to_brain(agent)

    for agent in protein_to_remove:
        if agent.uid in removed_ids:
            continue
        protein_name = agent.name
        Simulation.model.remove_agent(agent)
        removed_ids.add(agent.uid)
        Simulation.model.gut_add_cleaved_protein(protein_name)
        Simulation.model.gut_add_cleaved_protein(protein_name)

    Simulation.model.envs['gut']['context'].synchronize(restore_agent)

    for agent in all_true_cleaved_aggregates:
        if agent.uid in removed_ids:
            continue
        if agent.toAggregate and agent.is_valid():
            cont = 0
            _, agent_nghs_cleaved, _ = agent.check_and_get_nghs()
            for x in agent_nghs_cleaved:
                if x.alreadyAggregate and x.uid != agent.uid:
                    if cont < 3:
                        if Simulation.model.envs['gut']['context'].agent(x.uid) is not None:
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

    Simulation.model.envs['gut']['context'].synchronize(restore_agent)

    remove_agents = gather_agents_to_remove()
    for agent in remove_agents:
        if agent.uid not in removed_ids:
            if Simulation.model.envs['gut']['context'].agent(agent.uid) is not None:
                Simulation.model.remove_agent(agent)
                removed_ids.add(agent.uid)