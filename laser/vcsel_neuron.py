from vcsel_rate import VCSEL
from network_layer import NetworkLayer
class SpikingNeuron(NetworkLayer):
    """
    A wrapper defining a singular spiking neuron
    """
    def __init__(self, parameters, bias=(0, 0, 0), step_size=0.1):
        """
        inputs - a list of inputs that can be transferred 
        outputs - a list of outputs that can be transferred to other spiking neurons
        """
        self.laser = VCSEL(params=parameters, initial_state=bias) # define a laser item
        self.step_size = step_size
    def _time_range(self, times, epsilon = 10):
        """
        given a list of times -- times
        return (min_time - epsilon, max_time + epsilon)
        """
        minimum = min(times, key=lambda t: t[-1])[-1]
        maximum = max(times, key=lambda t: t[-1])[-1]
        return max(minimum - epsilon, 0), maximum + epsilon
    def forward(self, X):
        """
        Given a list of inputs at X, run a simulation
        """
        start, end = self._time_range(X, epsilon=5)
        self.laser.sim(start, end, self.step_size)
        return self.laser.sol

