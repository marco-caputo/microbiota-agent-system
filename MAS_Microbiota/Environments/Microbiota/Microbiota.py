from enum import IntEnum

from repast4py.space import DiscretePoint as dpt

from MAS_Microbiota import Simulation, restore_agent
from MAS_Microbiota.Environments import GridAgent, GridEnvironment, ResourceAgent
from MAS_Microbiota.Environments.Brain.Agents.Neurotransmitter import Neurotransmitter
from MAS_Microbiota.Environments.Brain.Agents.Precursor import Precursor, PrecursorType
from MAS_Microbiota.Environments.Microbiota.Agents import *


class Microbiota(GridEnvironment):

    NAME = 'microbiota'

    def __init__(self, context: GridEnvironment, grid: GridAgent):
        super().__init__(context, grid)
        self.substrates_to_add = {
            SubstrateType.FIBER: 0,
            SubstrateType.CARBOHYDRATE: 0,
            SubstrateType.SUGAR: 0
        }
        self.bacteria_to_add = []
        self.good_bacteria_count = 0
        self.pathogenic_bacteria_count = 0

    @staticmethod
    def initial_agents():
        return [
            ('enterobacteriaceae.count', Enterobacteriaceae, None),
            ('streptococcaceae.count', Streptococcaceae, None),
            ('clostridiaceae.count', Clostridiaceae, None),
            ('lactobacillaceae.count', Lactobacillaceae, None),
            ('prevotellaceae.count', Prevotellaceae, None),
            ('bifidobacteriaceae.count', Bifidobacteriaceae, None),
            ('ruminococcaceae.count', Ruminococcaceae, None),
            ('lachnospiraceae.count', Lachnospiraceae, None),

            ('scfa_acetate.count', SCFA, SCFAType.ACETATE),
            ('scfa_propionate.count', SCFA, SCFAType.PROPIONATE),
            ('scfa_butyrate.count', SCFA, SCFAType.BUTYRATE),
            ('substrate_fiber.count', Substrate, SubstrateType.FIBER),
            ('substrate_carbohydrate.count', Substrate, SubstrateType.CARBOHYDRATE),
            ('substrate_sugar.count', Substrate, SubstrateType.SUGAR),
            ('external_input_diet.count', ExternalInput, ExternalInputType.DIET),
            ('external_input_antibiotics.count', ExternalInput, ExternalInputType.ANTIBIOTICS),
            ('external_input_stress.count', ExternalInput, ExternalInputType.STRESS),
            ('treatment_diet.count', Treatment, TreatmentType.DIET),
            ('treatment_probiotics.count', Treatment, TreatmentType.PROBIOTICS),
        ]

    def step(self):
        removed_ids = set()
        self.context.synchronize(restore_agent)
        self.remove_agents(removed_ids)
        self.add_substrates()
        self.make_agents_steps()
        self.add_bacteria()
        resources_to_move = []

        for agent in self.context.agents():
            if isinstance(agent, ResourceAgent) and agent.toMove:
                resources_to_move.append(agent)
                agent.toRemove = True

        self.move_resources_to_brain(resources_to_move)
        self.remove_agents(removed_ids)
        self.apply_actions()
        self.remove_agents(removed_ids)
        self.count_bacteria()
        self.context.synchronize(restore_agent)
        self.remove_agents(removed_ids)


    def add_substrates(self):
        for type in list(SubstrateType):
            self.substrates_to_add[type] += Simulation.params["diet_substrates"]["intake"][type.name.lower()]

        for substrate_type in self.substrates_to_add:
            for _ in range(self.substrates_to_add[substrate_type]):
                pt = self.grid.get_random_local_pt(Simulation.model.rng)
                substrate = Substrate(Simulation.model.new_id(), Simulation.model.rank, substrate_type, pt, self.NAME)
                self.context.add(substrate)
            self.substrates_to_add[substrate_type] = min(self.substrates_to_add[substrate_type], 0)


    def move_resources_to_brain(self, resources_to_move):
        for agent in resources_to_move:
            if isinstance(agent, (SCFA, Precursor, Substrate)):
                Simulation.model.gutBrainInterface.transfer_to_bloodstream(agent)
            elif isinstance(agent, Neurotransmitter):
                Simulation.model.gutBrainInterface.transfer_to_enteric_nervous_system(agent)

    def count_bacteria(self):
        """
        Updates the count of good and pathogenic bacteria in the microbiota environment.
        """
        for bacterium in [b for b in self.context.agents() if isinstance(b, Bacterium)]:
            if bacterium.causes_inflammation():
                self.pathogenic_bacteria_count += 1
            else:
                self.good_bacteria_count += 1


    def apply_actions(self):
        for bacterium in [agent for agent in self.context.agents() if isinstance(agent, Bacterium)]: # For each bacterium in the context...
            bacterium.step() # Call the step method of the bacterium.
            if bacterium.toFission:
                self._fission(bacterium)
            for fermentable_type in bacterium.toFerment:
                if bacterium.toFerment[fermentable_type]:
                    self._ferment(bacterium, fermentable_type)

    def agents_to_remove(self):
        return Bacterium, SCFAType, Substrate, Precursor, Neurotransmitter

    def find_bact_free_nghs(self, pt: dpt) -> list[dpt]:
        """
        Finds the position around a given point that do not have any bacterium agent.
        :param pt: The point to check
        :return: A list of empty points
        """
        empty_pt_list = []
        nghs_coords = Simulation.model.ngh_finder.find(pt.x, pt.y)
        for ngh_coords in nghs_coords:
            nghs_agents = list(self.grid.get_agents(dpt(ngh_coords[0], ngh_coords[1])))
            if len([ag for ag in nghs_agents if isinstance(ag, Bacterium)]) == 0:
                 empty_pt_list.append(ngh_coords)
        return empty_pt_list

    def teleport_resources_step(self):
        for agent in [ag for ag in self.context.agents() if isinstance(ag, Substrate)]:
            pt = self.grid.get_random_local_pt(Simulation.model.rng)
            Simulation.model.move(agent, pt, agent.context)

    def _fission(self, bacterium):
        """
        Applies fission to a bacterium agent, creating a new bacterium agent in an empty position around the bacterium.
        If no empty position is found, the fission is not applied, but the state of the bacterium remains unchanged
        in order to try again in the next step.

        :param bacterium: The bacterium agent to apply fission
        """
        empty_ngh_pts = self.find_bact_free_nghs(bacterium.pt)
        if len(empty_ngh_pts) == 0:
            return
        point = Simulation.model.rng.choice(empty_ngh_pts)
        bact_class = type(bacterium)
        new_bacterium = bact_class(Simulation.model.new_id(), Simulation.model.rank, dpt(point[0], point[1]), self.NAME)
        self.context.add(new_bacterium)
        bacterium.toFission = False

    def _ferment(self, bacterium: Bacterium, fermentable_type: type[ResourceAgent]):
        neighbours = Simulation.model.ngh_finder.find(bacterium.pt.x, bacterium.pt.y)  # Neighbours of the bacterium...
        point = Simulation.model.rng.choice(neighbours)
        if fermentable_type == Substrate:
            self._add_metabolite(bacterium.produced_scfa(), SCFA, dpt(point[0], point[1]))
            self._add_metabolite(bacterium.produced_precursors(), Precursor, dpt(point[0], point[1]))
        elif fermentable_type == Precursor and bacterium.fermentedPrecursor != 0:
            fermented_precursor_type = PrecursorType(bacterium.fermentedPrecursor)
            self._add_metabolite(fermented_precursor_type.associated_neurotransmitters(), Neurotransmitter, dpt(point[0], point[1]))
            bacterium.fermentedPrecursor = 0

    # Based on the assumption that all metabolite and neurotransmitters agents have the same constructor signature.
    def _add_metabolite(self, types: list[IntEnum], agent_class: type, point: dpt):
        if len(types) > 0:
            type = types[0].__class__(Simulation.model.rng.choice(types))
            agent = agent_class(Simulation.model.new_id(), Simulation.model.rank, type, point, self.NAME)
            self.context.add(agent)

    def add_bacteria(self):
        for bacterium in self.bacteria_to_add:
            self.context.add(bacterium)
        self.bacteria_to_add = []