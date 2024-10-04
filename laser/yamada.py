import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import diffrax
from diffrax import diffeqsolve, ControlTerm, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree

import jax

class Yamada():
    """
    Generic Yamada model laser
    """
    def __init__(self, params: tuple, initial_state=(0, 0, 0), solver=diffrax.Tsit5()):
        self.params = params
        self.initial_state = initial_state
        self.solver = solver
        self.name = "yamada"  
    def simple_yamada(self, inputs, sim_step):
        def f(t, y, args):
            nonlocal inputs, sim_step
            G, Q, I = y
            A, B, a, y_G, y_Q, y_I = args
            d_G = y_G * (A - G - G * I)
            d_Q = y_Q * (B - Q - a * Q * I)
            d_I = y_I * (G - Q - 1) * I
            d_y = d_G, d_Q, d_I
            return d_y
        return f
    def sim(self, inputs, start_time, end_time, step):
        if (not inputs):
            inputs = jnp.zeros(shape=int((end_time-start_time)/step)) # inputs of zeros
        terms = ODETerm(self.simple_yamada(inputs, step))

        assert(isinstance(terms, ODETerm))
        assert(isinstance(self.initial_state, tuple))

        saveat = SaveAt(ts=jnp.linspace(start_time, end_time, 1000))
        
        sol = diffeqsolve(terms,
                          self.solver,
                          start_time,
                          end_time,
                          step,
                          self.initial_state,
                          args=self.params,
                          saveat=saveat)
        self.sol = sol
    def plot(self):
        sol = self.sol
        plt.plot(sol.ts, sol.ys[0], label="Gain")
        plt.plot(sol.ts, sol.ys[1], label="Absorption")
        plt.plot(sol.ts, sol.ys[2], label="Laser Intensity")
        plt.legend()

        plt.savefig(f"{self.name}_save.png")
        plt.show()

if __name__ == "__main__":
    yamada = Yamada(params=(0.1, 0.02, 0.4, 0.02, 0.5, 0.5), initial_state=(10., 10., 10.))
    yamada.sim(None, 0, 140, .1)
    yamada.plot()
