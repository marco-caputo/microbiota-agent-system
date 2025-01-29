from dataclasses import dataclass
from .Utils import Simulation
from MAS_Microbiota.Environments.Gut.Agents import *
from MAS_Microbiota.Environments.Brain.Agents import *
from MAS_Microbiota.Environments.Microbiota.Agents import *


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
    

    # Microbiota variables => new lines
    SCFA: int = 0
    precursor: int = 0


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
    # new variables
    dopamine: int = 0
    serotonin: int = 0
    norepinephrine: int = 0


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
            "tau_oligomer_brain": 0, 
            # new lines
            "SCFA": 0,
            "precursor": 0,
            "neurotransmitter": 0,
            "bacteria_good": 0,
            "bacteria_pathogenic": 0,
            "dopamine": 0,
            "serotonin": 0,
            "norepinephrine": 0
        }

        for env_name in Simulation.model.envs:
            context = Simulation.model.envs[env_name].context
            for agent in context.agents():
                if isinstance(agent, Oligomer):
                    if agent.name == ProteinName.ALPHA_SYN:
                        counts["alpha_oligomer_"+env_name] += 1
                    else:
                        counts["tau_oligomer_"+env_name] += 1
                elif isinstance(agent, CleavedProtein):
                    if agent.name == ProteinName.ALPHA_SYN:
                        counts["alpha_cleaved_"+env_name] += 1
                    else:
                        counts["tau_cleaved_"+env_name] += 1
                elif isinstance(agent, Neuron):
                    if agent.state == NeuronState.HEALTHY:
                        counts["neuron_healthy"] += 1
                    elif agent.state == NeuronState.DAMAGED:
                        counts["neuron_damaged"] += 1
                elif isinstance(agent, Microglia):
                    if agent.state == MicrogliaState.ACTIVE:
                        counts["microglia_active"] += 1
                    else:
                        counts["microglia_resting"] += 1
                elif (type(agent) == Protein):
                    if (agent.name == ProteinName.ALPHA_SYN):
                        counts["alpha_protein_"+env_name] += 1
                    else:
                        counts["tau_protein_"+env_name] += 1
                elif (type(agent) == AEP):
                    if (agent.state == AEPState.ACTIVE):
                        counts["aep_active"] += 1
                    else:
                        counts["aep_hyperactive"] += 1
                elif isinstance(agent, Neurotransmitter): ### new lines
                    if (agent.neurotrans_type == NeurotransmitterType.DOPAMINE):
                        counts["dopamine"] += 1
                    elif (agent.neurotrans_type == NeurotransmitterType.SEROTONIN):
                        counts["serotonin"] += 1
                    else:
                        counts["norepinephrine"] += 1
                elif isinstance(agent, Bacterium):
                    if agent.causes_inflammation():
                        counts["bacteria_pathogenic"] += 1
                    else:
                        counts["bacteria_good"] += 1
                elif isinstance(agent, SCFA):
                    if not agent.toRemove:
                        counts["SCFA"] += 1
                elif isinstance(agent, Precursor):
                    if not agent.toRemove:
                        counts["precursor"] += 1

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
        self.serotonin = counts["serotonin"]
        self.dopamine = counts["dopamine"]
        self.norepinephrine = counts["norepinephrine"]

        # gut
        self.aep_active = counts["aep_active"]
        self.aep_hyperactive = counts["aep_hyperactive"]
        self.alpha_protein_gut = counts["alpha_protein_gut"]
        self.tau_protein_gut = counts["tau_protein_gut"]
        self.alpha_cleaved_gut = counts["alpha_cleaved_gut"]
        self.tau_cleaved_gut = counts["tau_cleaved_gut"]
        self.alpha_oligomer_gut = counts["alpha_oligomer_gut"]
        self.tau_oligomer_gut = counts["tau_oligomer_gut"]
        self.microbiota_good_bacteria_class = Simulation.model.microbiota_good_bacteria_count
        self.microbiota_pathogenic_bacteria_class = Simulation.model.microbiota_pathogenic_bacteria_count
        self.microbiota_good_bacteria_class = counts["bacteria_good"]
        self.microbiota_pathogenic_bacteria_class = counts["bacteria_pathogenic"]
        self.SCFA = counts["SCFA"]
        self.precursor = counts["precursor"]

        
        self.barrier_impermeability = Simulation.model.epithelial_barrier_impermeability

    

        Simulation.model.data_set.log(tick)