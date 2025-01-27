from typing import Tuple
from repast4py.space import DiscretePoint as dpt

from MAS_Microbiota.Environments.Brain.Agents import *
from MAS_Microbiota.Environments.Gut.Agents import *
from MAS_Microbiota.Environments.Microbiota.Agents import *

agent_cache = {}

# Mapping agent types to their constructors and specific attributes
AGENT_MAPPING = {
    Microglia.TYPE: {
        "constructor": lambda uid, pt, context, data: Microglia(uid[0], uid[2], MicrogliaState(data[1]), pt, context),
        "attributes": lambda agent, data: setattr(agent, "state", MicrogliaState(data[1])),
    },
    Neuron.TYPE: {
        "constructor": lambda uid, pt, context, data: Neuron(uid[0], uid[2], pt, context),
        "attributes": lambda agent, data: (
            setattr(agent, "state", NeuronState(data[1])),
            setattr(agent, "toRemove", data[3])
        ),
    },
    CleavedProtein.TYPE: {
        "constructor": lambda uid, pt, context, data: CleavedProtein(uid[0], uid[2], ProteinName(data[1]), pt, context),
        "attributes": lambda agent, data: (
            setattr(agent, "toAggregate", data[3]),
            setattr(agent, "alreadyAggregate", data[4]),
            setattr(agent, "toRemove", data[5])
        ),
    },
    Oligomer.TYPE: {
        "constructor": lambda uid, pt, context, data: Oligomer(uid[0], uid[2], ProteinName(data[1]), pt, context),
        "attributes": lambda agent, data: (
            setattr(agent, "toRemove", data[3]),
        ),
    },
    Cytokine.TYPE: {
        "constructor": lambda uid, pt, context, data: Cytokine(uid[0], uid[2], pt, context),
        "attributes": lambda agent, data: setattr(agent, "state", MicrogliaState(data[1])),
    },
    AEP.TYPE: {
        "constructor": lambda uid, pt, context, data: AEP(uid[0], uid[2], pt, context),
        "attributes": lambda agent, data: setattr(agent, "state", AEPState(data[1])),
    },
    Protein.TYPE: {
        "constructor": lambda uid, pt, context, data: Protein(uid[0], uid[2], ProteinName(data[1]), pt, context),
        "attributes": lambda agent, data: (
            setattr(agent, "toCleave", data[3]),
            setattr(agent, "toRemove", data[4]),
        ),
    },
    ExternalInput.TYPE: {
        "constructor": lambda uid, pt, context, data: ExternalInput(uid[0], uid[2], ExternalInputType(data[1]), pt, context),
        "attributes": lambda agent, _: None,
    },
    Treatment.TYPE: {
        "constructor": lambda uid, pt, context, data: Treatment(uid[0], uid[2], TreatmentType(data[1]), pt, context),
        "attributes": lambda agent, _: None,
    },
    Bacterium.TYPE: {
        "constructor": lambda uid, pt, context, data: data[1](uid[0], uid[2], pt, context),
        "attributes": lambda agent, data: (
            setattr(agent, "toFission", data[4]),
            setattr(agent, "toFerment", {Substrate: data[5], Precursor: data[6]}),
            setattr(agent, "fermentedPrecursor", data[7]),
            setattr(agent, "toRemove", data[8]),
            setattr(agent, "energy_level", EnergyLevel(data[9])),
        ),
    },
    SCFA.TYPE: {
        "constructor": lambda uid, pt, context, data: SCFA(uid[0], uid[2], SCFAType(data[1]), pt, context),
        "attributes": lambda agent, data: (
            setattr(agent, "toRemove", data[4]),
        ),
    },
    Substrate.TYPE: {
        "constructor": lambda uid, pt, context, data: Substrate(uid[0], uid[2], SubstrateType(data[1]), pt, context),
        "attributes": lambda agent, data: (
            setattr(agent, "toRemove", data[4]),
        ),
    },
    Precursor.TYPE: {
        "constructor": lambda uid, pt, context, data: Precursor(uid[0], uid[2], PrecursorType(data[1]), pt, context),
        "attributes": lambda agent, data: (
            setattr(agent, "toRemove", data[4]),
        ),
    },
    Neurotransmitter.TYPE: {
        "constructor": lambda uid, pt, context, data: Neurotransmitter(uid[0], uid[2], NeurotransmitterType(data[1]), pt, context),
        "attributes": lambda agent, data: (
            setattr(agent, "toMove", data[4]),
            setattr(agent, "toRemove", data[5]),
        ),
    },
}

# Generalized restore function
def restore_agent(agent_data: Tuple):
    uid = agent_data[0]
    pt_array = agent_data[2]
    pt = dpt(pt_array[0], pt_array[1], 0)

    agent_type = uid[1]
    if agent_type not in AGENT_MAPPING:
        raise ValueError(f"Unknown agent type: {agent_type}")

    context = agent_data[3] if len(agent_data) > 3 else None
    if uid in agent_cache:
        agent = agent_cache[uid]
    else:
        constructor = AGENT_MAPPING[agent_type]["constructor"]
        agent = constructor(uid, pt, context, agent_data)
        agent_cache[uid] = agent

    # Apply type-specific attributes
    attributes = AGENT_MAPPING[agent_type]["attributes"]
    attributes(agent, agent_data)

    # Set common attributes
    agent.pt = pt
    if context is not None:
        agent.context = context

    return agent