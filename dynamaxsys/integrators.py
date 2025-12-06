import jax.numpy as jnp
from dynamaxsys.base import (
    LinearControlDynamics,
    LinearControlDisturbanceDynamics,
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
