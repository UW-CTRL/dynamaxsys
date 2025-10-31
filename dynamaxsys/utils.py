import jax
import equinox as eqx
import jax.numpy as jnp

@eqx.filter_jit
def runge_kutta_integrator(dynamics, dt=0.1):
    # zero-order hold
    def integrator(x, u, d=None, t=0.0):
        dt2 = dt / 2.0
        k1 = dynamics(x, u, d, t)
        k2 = dynamics(x + dt2 * k1, u, d, t + dt2)
        k3 = dynamics(x + dt2 * k2, u, d, t + dt2)
        k4 = dynamics(x + dt * k3, u, d, t + dt)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return integrator

@eqx.filter_jit
def linearize(dynamics, state, control, disturbance=None, time=0.0):
    if disturbance is None:
        disturbance = jnp.zeros((dynamics.disturbance_dim,))
    A, B, C = jax.jacobian(dynamics, [0, 1, 2])(state, control, disturbance, time)
    D = dynamics(state, control, disturbance, time) - A @ state - B @ control - C @ disturbance
    return A, B, C, D