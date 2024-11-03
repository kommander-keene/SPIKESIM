from vcsel_neuron import SpikingNeuron
from normalize_layer import Normalize
from vcsel_rate import default_vcsel_parameters
from vcsel_rate import format_inputs, get_vcsel_yamada_model
class SpikingNetwork():
    """
    A class defining a spiking neural network.
    """

    def __init__(self):
        self.network_layers = []

    def add_input(self, neuron):
        self.network_layers.append(neuron)
    def forward(self, input, numerical_outputs=False):
        forwarded_inputs = input
        for l in self.network_layers:
            solution = l.forward(forwarded_inputs)
            forwarded_inputs = solution # TODO this is probably busted
        if (numerical_outputs):
            return forwarded_inputs.ys
        return forwarded_inputs
            
def one_network():
    DEFAULTPARAMS = default_vcsel_parameters()
    A = 10
    spikes = [(A, 208), (-A, 833), (A, 1562.5), (A, 1666), (A, 1875), (A, 2083)]
    # format inputs for fun
    test_inputs = format_inputs(DEFAULTPARAMS, spikes)
    yamada_parameters = get_vcsel_yamada_model(DEFAULTPARAMS, spikes=test_inputs)

    neuron = SpikingNeuron(yamada_parameters)
    network = SpikingNetwork()
    network.add_input(neuron)
    network.add_input(Normalize(DEFAULTPARAMS))
    network.add_input(SpikingNeuron(yamada_parameters))
    network.add_input(Normalize(DEFAULTPARAMS))
    outputs = network.forward(test_inputs)
if __name__ == "__main__":
    # test a 1 neuron network
    one_network()
