from dataclasses import dataclass
from .Utils import Simulation
from MAS_Microbiota.Environments.Gut.Agents import *
from MAS_Microbiota.Environments.Brain.Agents import *


@dataclass
class Log:
    
    # Gut variables
    aep_active: int = 0
    aep_hyperactive: int = 0
    alpha_protein_gut: int = 0
    tau_protein_gut: int = 0
    alpha_cleaved_gut: int = 0
    tau_cleaved_gut: int = 0
    alpha_oligomer_gut: int = 0
    tau_oligomer_gut: int = 0
    barrier_impermeability : int = 0
    microbiota_good_bacteria_class : int = 0
    microbiota_pathogenic_bacteria_class : int = 0
    
    # Brain variables
    resting_microglia: int = 0
    active_microglia: int = 0
    healthy_neuron: int = 0
    damaged_neuron: int = 0
    dead_neuron: int = 0
    cleaved_alpha_syn_brain: int = 0
    alpha_syn_oligomer_brain: int = 0
    cleaved_tau_brain: int = 0
    tau_oligomer_brain: int = 0
    cytokine_pro_inflammatory: int = 0
    cytokine_non_inflammatory: int = 0

    # Function to log the counts of the agents
    def log_counts(self):
        tick = Simulation.model.runner.schedule.tick

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

        for env in Simulation.model.envs:
            context = Simulation.model.envs[env]["context"]
            for agent in context.agents():
                if isinstance(agent, Oligomer):
                    if agent.name == Simulation.params["protein_name"]["alpha_syn"]:
                        counts["alpha_oligomer_"+env] += 1
                    else:
                        counts["tau_oligomer_"+env] += 1
                elif isinstance(agent, CleavedProtein):
                    if agent.name == Simulation.params["protein_name"]["alpha_syn"]:
                        counts["alpha_cleaved_"+env] += 1
                    else:
                        counts["tau_cleaved_"+env] += 1
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
                elif (type(agent) == Protein):
                    if (agent.name == Simulation.params["protein_name"]["alpha_syn"]):
                        counts["alpha_protein_"+env] += 1
                    else:
                        counts["tau_protein_"+env] += 1
                elif (type(agent) == AEP):
                    if (agent.state == Simulation.params["aep_state"]["active"]):
                        counts["aep_active"] += 1
                    else:
                        counts["aep_hyperactive"] += 1

        # brain
        self.healthy_neuron = counts["neuron_healthy"]
        self.damaged_neuron = counts["neuron_damaged"]
        self.dead_neuron = Simulation.model.dead_neuron
        self.cytokine_pro_inflammatory = Simulation.model.pro_cytokine
        self.cytokine_non_inflammatory = Simulation.model.anti_cytokine
        self.cleaved_alpha_syn_brain = counts["alpha_cleaved_brain"]
        self.cleaved_tau_brain = counts["tau_cleaved_brain"]
        self.alpha_syn_oligomer_brain = counts["alpha_oligomer_brain"]
        self.tau_oligomer_brain = counts["tau_oligomer_brain"]
        self.resting_microglia = counts["microglia_resting"]
        self.active_microglia = counts["microglia_active"]

        # gut
        self.aep_active = counts["aep_active"]
        self.aep_hyperactive = counts["aep_hyperactive"]
        self.alpha_protein_gut = counts["alpha_protein_gut"]
        self.tau_protein_gut = counts["tau_protein_gut"]
        self.alpha_cleaved_gut = counts["alpha_cleaved_gut"]
        self.tau_cleaved_gut = counts["tau_cleaved_gut"]
        self.alpha_oligomer_gut = counts["alpha_oligomer_gut"]
        self.tau_oligomer_gut = counts["tau_oligomer_gut"]
        self.microbiota_good_bacteria_class = Simulation.model.microbiota_good_bacteria_class
        self.microbiota_pathogenic_bacteria_class = Simulation.model.microbiota_pathogenic_bacteria_class
        self.barrier_impermeability = Simulation.model.barrier_impermeability

        Simulation.model.data_set.log(tick)