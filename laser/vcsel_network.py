from vcsel_neuron import SpikingNeuron
from normalize_layer import Normalize
from vcsel_rate import default_vcsel_parameters
from vcsel_rate import format_inputs, get_vcsel_yamada_model, default_steady_state
class SpikingNetwork():
    """
    A class defining a spiking neural network.
    """
    def __init__(self):
        self.network_layers = []

    def add_input(self, neuron, name):
        self.network_layers.append(neuron)
        self.network_layers[-1].name = name
    def forward(self, input, numerical_outputs=False, debug_plot = True):
        forwarded_inputs = input
        for idx, l in enumerate(self.network_layers):
            solution = l.forward(forwarded_inputs)
            if (debug_plot):
                l.plot(name=f"layer_{idx}")
            forwarded_inputs = solution
        if (numerical_outputs):
            return forwarded_inputs.ys
        return forwarded_inputs
    
def one_network():
    DEFAULTPARAMS = default_vcsel_parameters()
    A = 1 # this seems to be OFF for some reason
    spikes = [(A, 208), (-A, 833), (A, 1562.5), (A, 1666), (A, 1875), (A, 2083)]
    # format inputs for fun
    test_inputs = spikes

    network = SpikingNetwork()
    network.add_input(SpikingNeuron(DEFAULTPARAMS, bias=default_steady_state()), name="neuron1")
    network.add_input(Normalize(1, DEFAULTPARAMS), name="normalizingLayer2")
    network.add_input(SpikingNeuron(DEFAULTPARAMS, bias=default_steady_state()), name="neuron2")
    outputs = network.forward(test_inputs)


if __name__ == "__main__":
    # test a 1 neuron network
    one_network()
