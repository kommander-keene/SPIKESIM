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
        print(scaling)
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
    
    # NOTE CONVERT BACK FROM G Q I to physical constants before plotting
if __name__ == "__main__":
    # TODO scaling
    A = 2e-15 # really small
    spikes = ((A, 208), (-A, 833), (A, 1562.5), (A, 1666), (A, 1875), (A, 2083))
    # spikes = ((0, 0),)
    output_spikes = VCSEL.inputs(default_vcsel_parameters(), spikes)
    # print(output_spikes)
    yamada_parameters = VCSEL.get_yamada(default_vcsel_parameters(), spikes=output_spikes)
    # print(yamada_parameters)
    laser = VCSEL(params=yamada_parameters, initial_state=(0, 0, 0.))
    laser.sim(0, 3125, .1)
    laser.plot()