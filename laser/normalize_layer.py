from network_layer import NetworkLayer
from vcsel_rate import delta, format_inputs

class Normalize(NetworkLayer):
    def __init__(self, vcsel_params):
        self.params = vcsel_params
    def forward(self, X, threshold = 100, A = 10.0):
        """
        Recieves a y's and extracts
        """
        intensity_output = X.ys[-1] # intensity
        times = X.ts # time step information
        outputs = []
        for e, i in enumerate(intensity_output):
            if (abs(i) > threshold):
                outputs.append((A*(1 if i > 0 else -1), float(times[e])))

        return format_inputs(self.params, outputs)