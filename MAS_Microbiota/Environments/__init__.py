from .Gut import CleavedProtein, Oligomer, AEP, Protein, ExternalInput, Treatment
from .Gut import gut_step, move_cleaved_protein_step, microbiota_dysbiosis_step
from .Brain import Microglia, Neuron, Cytokine
from .Brain import brain_step, brain_add_cleaved_protein
from .GutBrainInterface import GutBrainInterface