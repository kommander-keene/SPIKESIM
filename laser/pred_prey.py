import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import diffrax
from diffrax import diffeqsolve, ControlTerm, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree

import jax

class PredPreyEQ():
    """
    Predator-Prey testing class
    """
    def __init__(self, params: tuple, initial_state=(0, 0), solver=diffrax.Tsit5()):
        self.params = params
        self.initial_state = initial_state
        self.solver = solver
        self.name = "predprey"  
    def simple_yamada(self):
        def vector_field(t, y, args):
            prey, predator = y
            α, β, γ, δ = args
            d_prey = α * prey - β * prey * predator
            d_predator = -γ * predator + δ * prey * predator
            d_y = d_prey, d_predator
            return d_y
        def f(t, y, args):
            G, Q, I = y
            A, B, a, y_G, y_Q, y_I, _ = args
            d_G = y_G * (A - G - G * I)
            d_Q = y_Q * (B - Q - a * Q * I)
            d_I = y_I * (G - Q - 1) * I
            d_y = d_G, d_Q, d_I
            return d_y
        return vector_field
    def sim(self, start_time, end_time, step):
        terms = ODETerm(self.simple_yamada())
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
        plt.plot(sol.ts, sol.ys[0], label="Prey")
        plt.plot(sol.ts, sol.ys[1], label="Predator")
        plt.legend()

        plt.savefig(f"{self.name}_save.png")
        plt.show()

if __name__ == "__main__":
    yamada = PredPreyEQ(params=(0.1, 0.02, 0.4, 0.02), initial_state=(10., 10.))
    yamada.sim(0, 140, 0.1)
    yamada.plot()
