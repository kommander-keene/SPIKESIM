from network_layer import NetworkLayer
from vcsel_rate import delta, format_inputs
import matplotlib.pyplot as plt

class Normalize(NetworkLayer):
    def __init__(self, 
                 vcsel_params,
                 time_delay=100,
                 A = 0.1):
        self.params = vcsel_params
        self.time_delay = time_delay
        self.A = A
    def _time_range(self, times, epsilon = 10):
        """
        given a list of times -- times
        return (min_time - epsilon, max_time + epsilon)
        """
        minimum = min(times)
        maximum = max(times)
        return max(minimum - epsilon, 0), maximum + epsilon
    
    def forward(self, X):
        """
        Recieves a y's and extracts
        """
        intensity_output = X.ys[2] # intensity
        times = X.ts # time step information
        maximum, minimum = self._time_range(times, epsilon=0)
        outputs = [] # conserve original time frame
        for e, i in enumerate(intensity_output):
            outputs.append((self.A*i, float(times[e]) + self.time_delay))
        self.plotoutputs = format_inputs(self.params, outputs)
        return self.plotoutputs
    def plot(self, name="", parameters=None, spikes=None):
        fig, ax1= plt.subplots(1, 1, figsize=(6, 6))
        amplitudes = []
        times = []
        for (A, T) in self.plotoutputs:
            amplitudes.append(A)
            times.append(T)
        plt.plot(times, amplitudes)
        ax1.set_title('Normalized Inputs')
        ax1.set_xlabel('Time / Photon Lifetime')
        ax1.set_ylabel('a.u. ')
        plt.savefig(f"{name}_normalized_inputs.png")