import jax.numpy as jnp
import equinox as eqx

from dynamaxsys.base import ControlAffineDynamics, ControlDisturbanceAffineDynamics
from typing import Callable


class ParametricControlAffineDynamics(ControlAffineDynamics):
    """
    Dynamics for control-affine systems with parametric control bounds: \dot{x} = f(x, t) + G(x, t) alpha_m u.
    It will take in the drift dynamics f(x, t) and control Jacobian G(x, t) of the original system, and construct a new system with state [x, alpha_m] where alpha_m is the control bound parameter.


    Attributes:
        drift_dynamics: Function for the drift term f(x, t).
        control_jacobian: Function for the control Jacobian G(x, t).
        state_dim: State dimension.
        control_dim: Control input dimension.
        disturbance_dim: Disturbance dimension (default 0).

    Methods:
        open_loop_dynamics: Returns the drift dynamics for a given state and time.
    """

    drift_dynamics: Callable[[jnp.ndarray, float], jnp.ndarray]
    control_jacobian: Callable[[jnp.ndarray, float], jnp.ndarray]
    state_dim: int
    control_dim: int
    disturbance_dim: int = 0

    def __init__(
        self,
        drift_dynamics: Callable[[jnp.ndarray, float], jnp.ndarray],
        control_jacobian: Callable[[jnp.ndarray, float], jnp.ndarray],
        state_dim: int,
        control_dim: int,
        disturbance_dim: int = 0,
    ):

        def parametric_drift_dynamics(
            x: jnp.ndarray,
            t: float = 0.0,
        ) -> jnp.ndarray:
            state = x[:-control_dim]
            return jnp.concatenate([drift_dynamics(state, t), jnp.zeros(control_dim)])

        def parametric_control_jacobian(
            x: jnp.ndarray,
            t: float = 0.0,
        ) -> jnp.ndarray:
            state = x[:-control_dim]
            parameter = x[-control_dim:]
            scale = jnp.diag(parameter)
            return jnp.concatenate(
                [
                    control_jacobian(state, t) @ scale,
                    jnp.zeros((control_dim, control_dim)),
                ],
                axis=0,
            )

        super().__init__(
            parametric_drift_dynamics,
            parametric_control_jacobian,
            state_dim + control_dim,
            control_dim,
        )

    @eqx.filter_jit
    def open_loop_dynamics(self, state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
        return self.drift_dynamics(state, time)

    @classmethod
    def from_control_affine_dynamics(
        cls,
        system: ControlAffineDynamics,
    ) -> "ParametricControlAffineDynamics":
        return cls(
            drift_dynamics=system.drift_dynamics,
            control_jacobian=system.control_jacobian,
            state_dim=system.state_dim,
            control_dim=system.control_dim,
        )


class ParametricControlDisturbanceAffineDynamics(ControlDisturbanceAffineDynamics):
    """
    Dynamics for control- and disturbance-affine systems with parametric control bounds: \dot{x} = f(x, t) + G(x, t) alpha_m u + H(x, t) beta_m d.
    It will take in the drift dynamics f(x, t), control Jacobian G(x, t), and disturbance Jacobian H(x, t) of the original system, and construct a new system with state [x, alpha_m, beta_m] where alpha_m and beta_m are the control and disturbance bound parameters.

    Attributes:
        drift_dynamics: Function for the drift term f(x, t).
        control_jacobian: Function for the control Jacobian G(x, t).
        disturbance_jacobian: Function for the disturbance Jacobian H(x, t).
        state_dim: State dimension.
        control_dim: Control input dimension.
        disturbance_dim: Disturbance input dimension.

    Methods:
        open_loop_dynamics: Returns the drift dynamics for a given state and time.
    """

    drift_dynamics: Callable[[jnp.ndarray, float], jnp.ndarray]
    control_jacobian: Callable[[jnp.ndarray, float], jnp.ndarray]
    disturbance_jacobian: Callable[[jnp.ndarray, float], jnp.ndarray]
    state_dim: int
    control_dim: int
    disturbance_dim: int

    def __init__(
        self,
        drift_dynamics: Callable[[jnp.ndarray, float], jnp.ndarray],
        control_jacobian: Callable[[jnp.ndarray, float], jnp.ndarray],
        disturbance_jacobian: Callable[[jnp.ndarray, float], jnp.ndarray],
        state_dim: int,
        control_dim: int,
        disturbance_dim: int,   
    ):
        def parametric_drift_dynamics(
            x: jnp.ndarray,
            t: float = 0.0,
        ) -> jnp.ndarray:
            state = x[:-control_dim - disturbance_dim]
            return jnp.concatenate([drift_dynamics(state, t), jnp.zeros(control_dim), jnp.zeros(disturbance_dim)])

        def parametric_control_jacobian(
            x: jnp.ndarray,
            t: float = 0.0,
        ) -> jnp.ndarray:
            state = x[:-control_dim - disturbance_dim]
            parameter = x[-control_dim - disturbance_dim:-disturbance_dim]
            scale = jnp.diag(parameter)
            return jnp.concatenate(
                [
                    control_jacobian(state, t) @ scale,
                    jnp.zeros((control_dim, control_dim)),
                    jnp.zeros((disturbance_dim, control_dim)),
                ],
                axis=0,
            )
            
        def parametric_disturbance_jacobian(
            x: jnp.ndarray,
            t: float = 0.0,
        ) -> jnp.ndarray:
            state = x[:-control_dim - disturbance_dim]
            parameter = x[-disturbance_dim:]
            scale = jnp.diag(parameter)
            return jnp.concatenate(
                [
                    disturbance_jacobian(state, t) @ scale,
                    jnp.zeros((control_dim, disturbance_dim)),
                    jnp.zeros((disturbance_dim, disturbance_dim)),
                ],
                axis=0,
            )

        super().__init__(
            parametric_drift_dynamics,
            parametric_control_jacobian,
            parametric_disturbance_jacobian,
            state_dim + control_dim + disturbance_dim,
            control_dim,
            disturbance_dim,
        )

    @eqx.filter_jit
    def open_loop_dynamics(self, state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
        return self.drift_dynamics(state, time)

    @classmethod
    def from_control_disturbance_affine_dynamics(
        cls,
        system: ControlDisturbanceAffineDynamics,
    ) -> "ParametricControlDisturbanceAffineDynamics":
        return cls(
            drift_dynamics=system.drift_dynamics,
            control_jacobian=system.control_jacobian,
            disturbance_jacobian=system.disturbance_jacobian,
            state_dim=system.state_dim,
            control_dim=system.control_dim,
            disturbance_dim=system.disturbance_dim,
        )

