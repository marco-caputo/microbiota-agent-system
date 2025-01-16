from Agents import *
from MAS_Microbiota.Environments.Gut.Agents import *
from MAS_Microbiota import Simulation

# Brain steps
# Function to add a cleaved protein agent to the brain context
def brain_add_cleaved_protein(self):
    self.added_agents_id += 1
    possible_types = [Simulation.params["protein_name"]["alpha_syn"], Simulation.params["protein_name"]["tau"]]
    random_index = np.random.randint(0, len(possible_types))
    cleaved_protein_name = possible_types[random_index]
    pt = self.brain_grid.get_random_local_pt(self.rng)
    cleaved_protein = CleavedProtein(self.added_agents_id, self.rank, cleaved_protein_name, pt, 'brain')
    self.brain_context.add(cleaved_protein)
    self.move(cleaved_protein, cleaved_protein.pt, cleaved_protein.context)

# Brain steps
def brain_step(self):
    self.brain_context.synchronize(restore_agent_brain)

    def gather_agents_to_remove():
        return [agent for agent in self.brain_context.agents() if
                isinstance(agent, (Oligomer, CleavedProtein, Neuron)) and agent.toRemove]

    # Remove agents marked for removal
    remove_agents = gather_agents_to_remove()
    removed_ids = set()
    for agent in remove_agents:
        if self.brain_context.agent(agent.uid) is not None:
            self.remove_agent(agent)
            removed_ids.add(agent.uid)

    self.brain_context.synchronize(restore_agent_brain)

    # Let each agent perform its step
    for agent in self.brain_context.agents():
        agent.step()

    # Collect data and perform operations based on agent states
    oligomer_to_remove = []
    active_microglia = 0
    damaged_neuron = 0
    all_true_cleaved_aggregates = []

    for agent in self.brain_context.agents():
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
        self.add_cytokine()
    for _ in range(damaged_neuron):
        self.brain_add_cleaved_protein()
    for oligomer in oligomer_to_remove:
        if self.brain_context.agent(oligomer.uid) is not None:
            self.remove_agent(oligomer)
            removed_ids.add(oligomer.uid)

    self.brain_context.synchronize(restore_agent_brain)

    for agent in all_true_cleaved_aggregates:
        if agent.uid in removed_ids:
            continue
        if agent.toAggregate and agent.is_valid():
            cont = 0
            _, agent_nghs_cleaved, _ = agent.check_and_get_nghs()
            for x in agent_nghs_cleaved:
                if x.alreadyAggregate and x.uid != agent.uid:
                    if cont < 3:
                        if self.brain_context.agent(x.uid) is not None:
                            self.remove_agent(x)
                            removed_ids.add(x.uid)
                        cont += 1
                    else:
                        x.alreadyAggregate = False
                        x.toAggregate = False
                        cont += 1
            self.add_oligomer_protein(agent.name, agent.context)
            self.remove_agent(agent)
            removed_ids.add(agent.uid)

    self.brain_context.synchronize(restore_agent_brain)

    # Remove agents marked for removal after all processing
    remove_agents = gather_agents_to_remove()
    for agent in remove_agents:
        if agent.uid not in removed_ids:
            if self.brain_context.agent(agent.uid) is not None:
                self.remove_agent(agent)
                removed_ids.add(agent.uid)

# Function to add a cytokine agent to the brain context
def add_cytokine(self):
    self.added_agents_id += 1
    pt = self.brain_grid.get_random_local_pt(self.rng)
    cytokine = Cytokine(self.added_agents_id, self.rank, pt, 'brain')
    self.brain_context.add(cytokine)
    self.move(cytokine, cytokine.pt, 'brain')