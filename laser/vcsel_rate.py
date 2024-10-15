import jax.random as jr
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True) # due to smaller numbers
jax.config.update("jax_debug_nans", True)
import matplotlib.pyplot as plt
import diffrax
from diffrax import diffeqsolve, ODETerm, SaveAt
from yamada import Yamada

def default_vcsel_parameters():
    """
    VCSEL parameters copied from the paper
    """
    return (
        850e-9, # Lasing wavelength [m]
        2.4e-18, # gain region cavity vol [m^3]
        2.4e-18, # SA region cavity vol [m^3]
        0.06, # gain region confinement factor 
        0.05, # SA region confinement factor
        1e-9, # gain region carrier lifetime [s]
        100e-12, # SA region carrier lifetime [s]
        4.8e-12, # Photon lifetime [s]
        2.9e-12, # gain region differential gain/loss [m^3 s^-1]
        14.5e-12, # SA region differential gain/loss [m^3 s^-1]
        1.1e24, # gain region transparency carrier density [m^-3]
        0.89e24, # SA region transparency carrier density [m^-3]
        10e-16, # Bimolecular recombination term [m^3 s^-1]
        1e-4, # spontaneous emission coupling factor
        0.4, # output power coupling coefficient
        2e-3, # A gain biasing current [A]
        0, # S absorber biasing current [A]
        )
class VCSEL(Yamada):
    """
    Implements a VCSEL specific simulation model based off the variable conversions
    """
    def inputs(vcsel_parameters, inputs):
        """
        Format optical and electrical inputs to be passed onto the rest of the network
        optical and electrical inputs should be lists of (float, float), but will be concatenated into one list
        """
        tau_A = vcsel_parameters[5]
        tau_S = vcsel_parameters[6]
        tau_ph = vcsel_parameters[7]
        gamma_a = vcsel_parameters[3]
        gamma_s = vcsel_parameters[4]
        g_a = vcsel_parameters[8]
        g_s = vcsel_parameters[9]
        I_a = vcsel_parameters[-2]
        I_s = vcsel_parameters[-1]
        Va = vcsel_parameters[1]
        Vs = vcsel_parameters[2]
        n_oa = vcsel_parameters[10]
        n_os = vcsel_parameters[11]
        beta = vcsel_parameters[13]
        Br = vcsel_parameters[12]

        # TODO optical input component is coupled with gain!
        # reformat this class to support that
        new_inputs = []
        scaling = (tau_A*g_a)/(1.6e-19*Va)
        for pair in inputs:
            new_inputs.append((scaling*pair[0], pair[1]))
        return new_inputs
    def get_yamada(vcsel_parameters, spikes = [(0, 0)]):
        """
        Create Yamada Parameters given the VCSEL parameters
        """
        tau_A = vcsel_parameters[5]
        tau_S = vcsel_parameters[6]
        tau_ph = vcsel_parameters[7]
        gamma_a = vcsel_parameters[3]
        gamma_s = vcsel_parameters[4]
        g_a = vcsel_parameters[8]
        g_s = vcsel_parameters[9]
        I_a = vcsel_parameters[-2]
        I_s = vcsel_parameters[-1]
        Va = vcsel_parameters[1]
        Vs = vcsel_parameters[2]
        n_oa = vcsel_parameters[10]
        n_os = vcsel_parameters[11]
        beta = vcsel_parameters[13]
        Br = vcsel_parameters[12]
        # define conversion parameters
        A = tau_A * tau_ph * gamma_a * g_a * (I_a/(1.6e-19*Va) - n_oa/tau_A)
        B = tau_S * tau_ph * gamma_s * g_s * (n_os/tau_S - I_s/(1.6e-19*Vs))
        a = (tau_S * gamma_s * g_s * Va) / (tau_A * gamma_a * g_a * Vs)
        y_G = tau_ph/tau_A
        y_Q = tau_ph/tau_S
        y_I = 1
        # epsilon_F approximated to most significant value
        epsilon_F = 8.35e-35
        # spikes (ASSUME PROCESSED)
        return A, B, a, y_G, y_Q, y_I, epsilon_F, spikes
    
    def __init__(self, params: tuple, initial_state=(0, 0, 0), solver=diffrax.Kvaerno5(), radius=0.001):
        super().__init__(params=params, initial_state=initial_state, solver=solver, radius=radius)
        self.name = "vcsel"
    
    def simple_yamada(self):
        def delta(t):        
            return jnp.exp(-jnp.pow(t, 2)/(2*self.spike_radius))
        def force(t, args):
            A, B, a, y_G, y_Q, y_I, eps, spikes = args
            spike = 0.0
            if (spikes and len(spikes) == 0):
                return 0.0
            for A, i in spikes:
                spike += A*delta(t - i)
            return spike
        def f(t, y, args):
            G, Q, I = y
            A, B, a, y_G, y_Q, y_I, eps, _ = args
            d_G = y_G * (A - G - G * I) + force(t, args)
            d_Q = y_Q * (B - Q - a * Q * I)
            d_I = y_I * (G - Q - 1) * I + eps*G*G
            d_y = d_G, d_Q, d_I
            return d_y
        return f
    def sim(self, start_time, end_time, step):
        terms = ODETerm(self.simple_yamada())
        assert(isinstance(terms, ODETerm))
        assert(isinstance(self.initial_state, tuple))
        saveat = SaveAt(ts=jnp.linspace(start_time, end_time, 1000))
        stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8, dtmax=0.1)
        sol = diffeqsolve(terms,
                          self.solver,
                          start_time,
                          end_time,
                          step,
                          self.initial_state,
                          args=self.params,
                          saveat=saveat,
                          max_steps=100000,
                          stepsize_controller=stepsize_controller)
        self.sol = sol
    def plot(self, vc_sel_parameters=None):
        sol = self.sol
        if (vc_sel_parameters):
            tau_A = vc_sel_parameters[5]
            tau_S = vc_sel_parameters[6]
            tau_ph = vc_sel_parameters[7]
            gamma_a = vc_sel_parameters[3]
            gamma_s = vc_sel_parameters[4]
            g_a = vc_sel_parameters[8]
            g_s = vc_sel_parameters[9]
            I_a = vc_sel_parameters[-2]
            I_s = vc_sel_parameters[-1]
            Va = vc_sel_parameters[1]
            Vs = vc_sel_parameters[2]
            n_oa = vc_sel_parameters[10]
            n_os = vc_sel_parameters[11]
            beta = vc_sel_parameters[13]
            Br = vc_sel_parameters[12]
            nu = vc_sel_parameters[14]
            lambda_ = vc_sel_parameters[0]
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6))
            ax1.plot(sol.ts, sol.ys[0]/(tau_ph * gamma_a * g_a) + n_oa, label="Gain Carrier Concentration", color="red")
            ax1.set_title('Gain Carrier Concentration')
            ax1.set_xlabel('Time / Photon Lifetime')
            ax1.set_ylabel('Density m^-3')

            ax2.plot(sol.ts, -(sol.ys[1]/(tau_ph * gamma_s * g_s) - n_oa), label="SA Carrier Concentration", color="blue")
            ax2.set_title('SA Carrier Concentration')
            ax2.set_xlabel('Time / Photon Lifetime')
            ax2.set_ylabel('Density m^-3')

            ax3.plot(sol.ts, (sol.ys[2]*Va/(tau_A * gamma_a * g_a))*(nu*gamma_a*1.98e-25)/(tau_ph*lambda_), label="Laser Output Power", color="green")
            ax3.set_title('Output Power')
            ax3.set_xlabel('Time / Photon Lifetime')
            ax3.set_ylabel('Density m^-3')

            # Adjust layout
            plt.tight_layout()
            plt.savefig(f"{self.name}_save_quantities.png")
            plt.show()
        else:
            plt.plot(sol.ts, sol.ys[0], label="Gain")
            plt.plot(sol.ts, sol.ys[1], label="Absorption")
            plt.plot(sol.ts, sol.ys[2], label="Laser Intensity")
            plt.legend()

            plt.savefig(f"{self.name}_save_variables.png")
            plt.show()
def experiment_1():
    # NOTE spiking power is arbitrary and unclear right now and noise is virtually ignored
    A = 2e-15
    spikes = ((A, 208), (-A, 833), (A, 1562.5), (A, 1666), (A, 1875), (A, 2083))
    # spikes = ((0, 0),)
    output_spikes = VCSEL.inputs(default_vcsel_parameters(), spikes)
    # print(output_spikes)
    yamada_parameters = VCSEL.get_yamada(default_vcsel_parameters(), spikes=output_spikes)
    # print(yamada_parameters)
    laser = VCSEL(params=yamada_parameters, initial_state=(4, 3, 0.))
    laser.sim(0, 3125, .1)
    laser.plot(default_vcsel_parameters())
def experiment_2():
    A = 2e-15
    spikes = ((-A, 260), (-A, 900), (-A, 1562), (A, 520), (A, 1041), (A, 1562), (A, 2083), (A, 2604), )
    # spikes = ((0, 0),)
    output_spikes = VCSEL.inputs(default_vcsel_parameters(), spikes)
    # print(output_spikes)
    yamada_parameters = VCSEL.get_yamada(default_vcsel_parameters(), spikes=output_spikes)
    # print(yamada_parameters)
    laser = VCSEL(params=yamada_parameters, initial_state=(4, 3, 0.))
    laser.sim(0, 3125, .1)
    laser.plot(None)
if __name__ == "__main__":
    experiment_1()