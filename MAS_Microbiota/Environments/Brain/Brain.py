import numpy as np

from .Agents import *
from MAS_Microbiota.Environments.Gut.Agents import *
from MAS_Microbiota.AgentRestorer import restore_agent
from MAS_Microbiota import Simulation

# Brain steps
# Function to add a cleaved protein agent to the brain context
def brain_add_cleaved_protein():
    Simulation.model.added_agents_id += 1
    possible_types = [Simulation.params["protein_name"]["alpha_syn"], Simulation.params["protein_name"]["tau"]]
    random_index = np.random.randint(0, len(possible_types))
    cleaved_protein_name = possible_types[random_index]
    pt = Simulation.model.brain_grid.get_random_local_pt(Simulation.model.rng)
    cleaved_protein = CleavedProtein(Simulation.model.added_agents_id, Simulation.model.rank, cleaved_protein_name, pt, 'brain')
    Simulation.model.brain_context.add(cleaved_protein)
    Simulation.model.move(cleaved_protein, cleaved_protein.pt, cleaved_protein.context)

# Brain steps
def brain_step():
    Simulation.model.brain_context.synchronize(restore_agent)

    def gather_agents_to_remove():
        return [agent for agent in Simulation.model.brain_context.agents() if
                isinstance(agent, (Oligomer, CleavedProtein, Neuron)) and agent.toRemove]

    # Remove agents marked for removal
    remove_agents = gather_agents_to_remove()
    removed_ids = set()
    for agent in remove_agents:
        if Simulation.model.brain_context.agent(agent.uid) is not None:
            Simulation.model.remove_agent(agent)
            removed_ids.add(agent.uid)

    Simulation.model.brain_context.synchronize(restore_agent)

    # Let each agent perform its step
    for agent in Simulation.model.brain_context.agents():
        agent.step()

    # Collect data and perform operations based on agent states
    oligomer_to_remove = []
    active_microglia = 0
    damaged_neuron = 0
    all_true_cleaved_aggregates = []

    for agent in Simulation.model.brain_context.agents():
        if isinstance(agent, Oligomer) and agent.toRemove:
            oligomer_to_remove.append(agent)
        elif isinstance(agent, Microglia) and agent.state == Simulation.params["microglia_state"]["active"]:
            active_microglia += 1
        elif isinstance(agent, Neuron) and agent.state == Simulation.params["neuron_state"]["damaged"]:
            damaged_neuron += 1
        elif isinstance(agent, CleavedProtein) and agent.toAggregate:
            all_true_cleaved_aggregates.append(agent)
            agent.toRemove = True

    for _ in range(active_microglia):
        add_cytokine()
    for _ in range(damaged_neuron):
        brain_add_cleaved_protein()
    for oligomer in oligomer_to_remove:
        if Simulation.model.brain_context.agent(oligomer.uid) is not None:
            Simulation.model.remove_agent(oligomer)
            removed_ids.add(oligomer.uid)

    Simulation.model.brain_context.synchronize(restore_agent)

    for agent in all_true_cleaved_aggregates:
        if agent.uid in removed_ids:
            continue
        if agent.toAggregate and agent.is_valid():
            cont = 0
            _, agent_nghs_cleaved, _ = agent.check_and_get_nghs()
            for x in agent_nghs_cleaved:
                if x.alreadyAggregate and x.uid != agent.uid:
                    if cont < 3:
                        if Simulation.model.brain_context.agent(x.uid) is not None:
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

    Simulation.model.brain_context.synchronize(restore_agent)

    # Remove agents marked for removal after all processing
    remove_agents = gather_agents_to_remove()
    for agent in remove_agents:
        if agent.uid not in removed_ids:
            if Simulation.model.brain_context.agent(agent.uid) is not None:
                Simulation.model.remove_agent(agent)
                removed_ids.add(agent.uid)

# Function to add a cytokine agent to the brain context
def add_cytokine():
    Simulation.model.added_agents_id += 1
    pt = Simulation.model.brain_grid.get_random_local_pt(Simulation.model.rng)
    cytokine = Cytokine(Simulation.model.added_agents_id, Simulation.model.rank, pt, 'brain')
    Simulation.model.brain_context.add(cytokine)
    Simulation.model.move(cytokine, cytokine.pt, 'brain')