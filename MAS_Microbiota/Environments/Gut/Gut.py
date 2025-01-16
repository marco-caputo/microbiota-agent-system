from Agents import *
from MAS_Microbiota import Simulation

# Gut steps
# Function to move the cleaved protein agents
def move_cleaved_protein_step(self):
    for agent in self.gut_context.agents():
        if type(agent) == CleavedProtein:
            if agent.alreadyAggregate == False:
                pt = self.gut_grid.get_random_local_pt(self.rng)
                self.move(agent, pt, agent.context)
    for agent in self.brain_context.agents():
        if type(agent) == CleavedProtein:
            if agent.alreadyAggregate == False:
                pt = self.brain_grid.get_random_local_pt(self.rng)
                self.move(agent, pt, agent.context)

# Function to check if the microbiota is dysbiotic and adjust the barrier impermeability
def microbiota_dysbiosis_step(self):
    if self.microbiota_good_bacteria_class - self.microbiota_pathogenic_bacteria_class <= self.microbiota_diversity_threshold:
        value_decreased = int((Simulation.params["barrier_impermeability"] * np.random.randint(0, 6)) / 100)
        if self.barrier_impermeability - value_decreased <= 0:
            self.barrier_impermeability = 0
        else:
            self.barrier_impermeability = self.barrier_impermeability - value_decreased
        number_of_aep_to_hyperactivate = value_decreased
        cont = 0
        for agent in self.gut_context.agents(agent_type=0):
            if agent.state == Simulation.params["aep_state"]["active"] and cont < number_of_aep_to_hyperactivate:
                agent.state = Simulation.params["aep_state"]["hyperactive"]
                cont += 1
            elif cont == number_of_aep_to_hyperactivate:
                break
    else:
        if self.barrier_impermeability < Simulation.params["barrier_impermeability"]:
            value_increased = int((Simulation.params["barrier_impermeability"] * np.random.randint(0, 4)) / 100)
            if (self.barrier_impermeability + value_increased) <= Simulation.params["barrier_impermeability"]:
                self.barrier_impermeability = self.barrier_impermeability + value_increased

def gut_step(self):
    self.gut_context.synchronize(restore_agent_gut)

    def gather_agents_to_remove():
        return [agent for agent in self.gut_context.agents() if
                isinstance(agent, (Oligomer, CleavedProtein, Protein)) and agent.toRemove]

    remove_agents = gather_agents_to_remove()
    removed_ids = set()
    for agent in remove_agents:
        if self.gut_context.agent(agent.uid) is not None:
            self.remove_agent(agent)
            removed_ids.add(agent.uid)

    self.gut_context.synchronize(restore_agent_gut)

    for agent in self.gut_context.agents():
        agent.step()

    protein_to_remove = []
    all_true_cleaved_aggregates = []
    oligomers_to_move = []

    for agent in self.gut_context.agents():
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
        self.gutBrainInterface.transfer_from_gut_to_brain(agent)

    for agent in protein_to_remove:
        if agent.uid in removed_ids:
            continue
        protein_name = agent.name
        self.remove_agent(agent)
        removed_ids.add(agent.uid)
        self.gut_add_cleaved_protein(protein_name)
        self.gut_add_cleaved_protein(protein_name)

    self.gut_context.synchronize(restore_agent_gut)

    for agent in all_true_cleaved_aggregates:
        if agent.uid in removed_ids:
            continue
        if agent.toAggregate and agent.is_valid():
            cont = 0
            _, agent_nghs_cleaved, _ = agent.check_and_get_nghs()
            for x in agent_nghs_cleaved:
                if x.alreadyAggregate and x.uid != agent.uid:
                    if cont < 3:
                        if self.gut_context.agent(x.uid) is not None:
                            self.remove_agent(x)
                            removed_ids.add(x.uid)
                        cont += 1
                    else:
                        x.alreadyAggregate = False
                        x.toAggregate = False
                        cont += 1
            self.add_oligomer_protein(agent.name, 'gut')
            self.remove_agent(agent)
            removed_ids.add(agent.uid)

    self.gut_context.synchronize(restore_agent_gut)

    remove_agents = gather_agents_to_remove()
    for agent in remove_agents:
        if agent.uid not in removed_ids:
            if self.gut_context.agent(agent.uid) is not None:
                self.remove_agent(agent)
                removed_ids.add(agent.uid)