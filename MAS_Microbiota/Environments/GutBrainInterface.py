from repast4py import random
from MAS_Microbiota import Simulation
from MAS_Microbiota.Environments import GridAgent
from MAS_Microbiota.Environments.Brain.Agents import Precursor, Neurotransmitter, Neuron
from MAS_Microbiota.Environments.Brain.Brain import Brain
from MAS_Microbiota.Environments.Gut.Agents import Oligomer
from MAS_Microbiota.Environments.Microbiota.Agents import SCFA


class GutBrainInterface:
    """
    Class that represents the interface between the gut and the brain. It acts as a unidirectional channel
    representing the bloodstream, the Blood-Brain Barrier, the enteric nervous system, and the vagus nerve.
    This class is responsible for:

    1. Transferring agents from the gut to the brain through the bloodstream, and so for updating the
    impermeability of the Blood-Brain Barrier based on the agents that pass through it.

    2.Receive neurotransmitters from the gut and make them influence the neurotransmitters production rates in the
    brain through the enteric nervous system and the vagus nerve.
    """

    def __init__(self, envs: dict):
        self.envs = envs
        self.bbb_impermeability = Simulation.params['blood_brain_barrier']['initial_impermeability']


    def transfer_to_bloodstream(self, agent: Oligomer | SCFA | Precursor):
        """
        Transfers the given agent from the gut to the brain through the bloodstream.
        Depending on the agent type, which should be one among Oligomer, SCFA and Precursor, the gut-brain interface
        can, respectively, transfer the agent to the brain, update the impermeability of the Blood-Brain Barrier, or
        transfer the agent to the brain if it passes through the BBB.

        parameters
        ---------- 
        agent: 
            The agent to be transferred to the brain.

        return
        ----------
        None
        
        """
        if isinstance(agent, Oligomer):
            self._transfer_to_brain(agent)
        elif isinstance(agent, SCFA):
            self._update_bbb_impermeability(agent)
        elif isinstance(agent, Precursor):
            if self._passes_through_bbb():
                self._transfer_to_brain(agent)


    def transfer_to_enteric_nervous_system(self, neurotrans: Neurotransmitter):
        """
        Transfers the given neurotransmitter agent from the gut to the enteric nervous system and
        applies the influence of the neurotransmitter on the production rates of neurotransmitters in the brain.
        A single neurotransmitter agent can influence one random neuron in the brain.

        :param neurotrans: The neurotransmitter agent to be transferred to the enteric nervous system.
        """
        neurons = [n for n in Simulation.model.envs[Brain.NAME].context.agents() if isinstance(n, Neuron)]
        if len(neurons) > 0:
            original_env_name = neurotrans.context
            neuron = Simulation.model.rng.choice(neurons)
            neuron.neurotrans_availability[neurotrans.neurotrans_type] += Simulation.params['neurotrans_rate_increase']
            neurotrans.toRemove = False
            neurotrans.toMove = False
            self.envs[original_env_name].remove(neurotrans)


    def _transfer_to_brain(self, agent: GridAgent):
        """
        Effectively transfers a given agent from its original environment to the brain.
        The agent is removed from the original environment and placed in a random location in
        the brain.

        :param agent: The agent to be transferred to the brain.
        """
        original_env_name = agent.context
        self.envs[Brain.NAME].context.add(agent)
        pt = self.envs[Brain.NAME].grid.get_random_local_pt(Simulation.model.rng)
        Simulation.model.move(agent, pt, Brain.NAME)
        agent.context = Brain.NAME
        agent.toRemove = False
        agent.toMove = False
        self.envs[original_env_name].remove(agent)

    def _update_bbb_impermeability(self, scfa: SCFA):
        """
        Updates the impermeability of the blood-brain barrier based on the SCFA agent.
        The impermeability is updated based on the SCFA type, hence its BBB impermeability coefficient,
        and the parameter that defines the influence of SCFA permeability on the BBB.
        In any case, the impermeability is limited to the minimum and maximum values defined in the
        parameters.

        :param scfa: The SCFA agent that will be used to update the impermeability.
        """
        self.bbb_impermeability += (scfa.BBB_impermeability_coefficient() *
                                    Simulation.params['blood_brain_barrier']['scfa_permeability_influence'])
        self.bbb_impermeability = max(Simulation.params['blood_brain_barrier']['minimum_impermeability'],
                                      min(Simulation.params['blood_brain_barrier']['maximum_impermeability'],
                                          self.bbb_impermeability)
                                      )
        scfa.toRemove = False
        scfa.toMove = False
        self.envs[scfa.context].remove(scfa)

    def _passes_through_bbb(self) -> bool:
        """
        Check to determine if a Precursor agent can pass through the Blood-Brain Barrier.
        This is determined by the current impermeability of the BBB and a random check.
        """
        return Simulation.model.rng.integers(0, 100) > self.bbb_impermeability