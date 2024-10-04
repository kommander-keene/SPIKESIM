import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import diffrax
from diffrax import diffeqsolve, ControlTerm, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree
from yamada_params import YamadaParams

class YamadaEquations():
    """
    Generically implements the Yamada laser equation models.
    """
    def __init__(self, params: YamadaParams, initial_state=jnp.array([0, 0, 0]), solver=diffrax.Tsit5()):
        """
        Set up equations here:

        dG/dt = y_G (A - G(t) - G(t) I(t))
        dQ/dt = y_Q (B - Q(t) - aQ(t)I(t))
        dI/dt = y_I (G(t) - Q(t) - 1) I(t) + eps * f(G)

        A - Bias current
        B - Absorption level
        a - differential absorption relative to maximum gain
        y_G - relaxation rate of the gain
        y_Q - relaxation rate of the absorber
        y_I - reverse photon lifetime 
        """
        self.params = params
        self.initial_state = initial_state
        self.solver = solver
        self.name = "yamada"
        
    def yamada_eq_terms(self, t0, t1, no_noise=True):
        """
        Defines a noiseless coupled set of Yamada diffEQs

        y = variables of interest
        args = arguments given (not used)
        """
        # individual equations
        def dGdt(t, y, args):
            G, Q, I = y
            A, B, a, y_G, y_Q, y_I, _ = args
            d_G = y_G * (A - G - G * I)
            return d_G
        def dQdt(t, y, args):
            G, Q, I = y
            A, B, a, y_G, y_Q, y_I, _ = args
            d_Q = y_Q * (B - Q - a * Q * I)
            return d_Q
        def dIdt(t, y, args):
            G, Q, I = y
            A, B, a, y_G, y_Q, y_I, _ = args
            d_I = y_I * (G - Q - 1) * I
            return d_I
        def noisyTerm(t, y, args):
            G, Q, I = y
            A, B, a, y_G, y_Q, y_I, eps = args
            return eps
        # term_to_noise
        brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=jr.PRNGKey(0)) # standin for noise
        terms = None
        if (no_noise):
            terms = ODETerm((dGdt, dQdt, dIdt))
        else:
            combined_terms = MultiTerm(ODETerm(dIdt), ControlTerm(noisyTerm, brownian_motion))
            deterministic_terms = (ODETerm(dGdt), ODETerm(dQdt)) # TODO does this work?
            terms = MultiTerm(deterministic_terms[0], deterministic_terms[1], combined_terms)
        return terms
    def sim(self, start_time, end_time, step):
        terms = self.yamada_eq_terms(start_time, end_time, True)
        print(terms)
        saveat = SaveAt(ts=jnp.linspace(start_time, end_time, 1000))
        sol = diffeqsolve(terms,
                          self.solver,
                          start_time,
                          end_time,
                          step,
                          self.initial_state,
                          args=self.params.to_args(),
                          saveat=saveat)
        self.sol = sol
    def plot(self):
        sol = self.sol
        plt.plot(sol.ts, sol.ys[0], label="Gain")
        plt.plot(sol.ts, sol.ys[1], label="Absorption")
        plt.plot(sol.ts, sol.ys[1], label="Laser Intensity")
        plt.legend()

        plt.savefig(f"{self.name}_save.png")
        plt.show()

if __name__ == "__main__":
    yamada_parameters = YamadaParams(1, 2, 0.4, 0.01, 0.01, 0.01)
    yamada = YamadaEquations(params=yamada_parameters, initial_state=jnp.array([1, 1, 1]))
    yamada.sim(0, 140, 0.1)
    yamada.plot()
