from network_layer import NetworkLayer
from vcsel_rate import delta, format_inputs
import matplotlib.pyplot as plt

class Normalize(NetworkLayer):
    def __init__(self, 
                 photodiode_R,
                 vcsel_params,
                 time_delay=1000,
                 A = 10):
        self.params = vcsel_params
        self.time_delay = time_delay
        self.A = A
        self.R = photodiode_R
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
        Recieves a y's and performs weighting while 
        """
        intensity_output = X.ys[2] # intensity
        times = X.ts # time step information
        maximum, minimum = self._time_range(times, epsilon=0)
        outputs = [] # conserve original time frame

        def to_power(Nph):
            # uses conversion
            tau_A = self.params[5]
            tau_ph = self.params[7]
            gamma_a = self.params[3]
            g_a = self.params[8]
            Va = self.params[1]
            nu = self.params[14]
            lambda_ = self.params[0]

            return (Nph*Va/(tau_A * gamma_a * g_a))*(nu*gamma_a*1.98e-25)/(tau_ph*lambda_)
        for e, i in enumerate(intensity_output):
            # takes intensity and converts it to current using photodiode R*P_in = I_ph
            outputs.append((self.A*self.R*to_power(i), float(times[e]) + self.time_delay))
        self.plotoutputs = format_inputs(self.params, outputs)
        return self.plotoutputs
    def plot(self, name="", parameters=None, spikes=None):
        fig, ax1= plt.subplots(1, 1, figsize=(6, 6))
        amplitudes = []
        times = []
        for (A, T) in self.plotoutputs:
            amplitudes.append(A)
            times.append(T)
        ax1.plot(times, amplitudes)
        ax1.set_title('Normalized Inputs')
        ax1.set_xlabel('Time / Photon Lifetime')
        ax1.set_ylabel('Amps ')
        fig.savefig(f"{name}_normalized_inputs.png")