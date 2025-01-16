from repast4py import parameters

"""
This class is used to store global parameters and the model instance. 
It is used to avoid passing the model instance  to every agent and to avoid passing the parameters to every agent.

Accessible parameters:
- params: A dictionary of global parameters
- model: The model instance
"""
class Simulation:
    params = {}
    model = None

    # Load parameters from the args file and store them in the class for easy global access
    @classmethod
    def load_from_args(cls):
        parser = parameters.create_args_parser()
        args = parser.parse_args()
        cls.params = parameters.init_params(args.parameters_file, args.parameters)

    @classmethod
    def set_model(cls, model):
        cls.model = model