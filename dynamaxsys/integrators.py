import jax.numpy as jnp
from dynamaxsys.base import (
    LinearControlDynamics,
    LinearControlDisturbanceDynamics,
    ControlAffineDynamics,
)


class IntegratorND(LinearControlDynamics):
    integrator_dim: int
    N_dim: int

    def __init__(self, integrator_dim, N_dim):
        self.integrator_dim = integrator_dim
        self.N_dim = N_dim
        state_dim = self.integrator_dim * self.N_dim
        control_dim = self.N_dim

        A = jnp.eye(state_dim, k=self.N_dim)
        B = jnp.zeros([state_dim, control_dim])
        B = B.at[-self.N_dim :].set(jnp.eye(self.N_dim))

        super().__init__(A, B)


def DoubleIntegrator2D():
    return IntegratorND(2, 2)


def DoubleIntegrator1D():
    return IntegratorND(2, 1)


def SingleIntegrator2D():
    return IntegratorND(1, 2)


def SingleIntegrator1D():
    return IntegratorND(1, 1)


class TwoPlayerRelativeIntegratorND(LinearControlDisturbanceDynamics):
    integrator_dim: int
    N_dim: int

    def __init__(self, integrator_dim, N_dim):
        self.integrator_dim = integrator_dim
        self.N_dim = N_dim
        state_dim = self.integrator_dim * self.N_dim
        # control_dim = self.N_dim * 2

        A = jnp.eye(state_dim, k=self.N_dim)
        B = jnp.zeros([state_dim, self.N_dim])
        B = B.at[-self.N_dim :].set(jnp.eye(self.N_dim))
        super().__init__(A, -B, B)


class ParametricIntegratorND(ControlAffineDynamics):
    integrator_dim: int
    N_dim: int

    def __init__(self, integrator_dim, N_dim):
        self.integrator_dim = integrator_dim
        self.N_dim = N_dim
        state_dim = self.integrator_dim * self.N_dim
        control_dim = self.N_dim

        A = jnp.eye(state_dim, k=self.N_dim)
        A_parameter = jnp.concatenate(
            [
                jnp.concatenate([A, jnp.zeros([1, state_dim])], axis=0),
                jnp.zeros([state_dim + 1, 1]),
            ],
            axis=1,
        )
        B = jnp.zeros([state_dim, control_dim])
        B = B.at[-self.N_dim :].set(jnp.eye(self.N_dim))
        B_parameter = jnp.concatenate([B, jnp.zeros([1, control_dim])], axis=0)

        def drift_dynamics(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            return A_parameter @ state

        def control_jacobian(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            alpha = state[-1]
            return B_parameter * alpha

        super().__init__(drift_dynamics, control_jacobian, state_dim + 1, control_dim)
