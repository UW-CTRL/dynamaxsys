from dynamaxsys.utils import runge_kutta_integrator, linearize
from typing import Callable, Union
import jax.numpy as jnp
import equinox as eqx


class Dynamics(eqx.Module):
    """
    Base class for continuous-time system dynamics.
    
    Attributes:
        dynamics_func: Callable implementing the system dynamics.
        state_dim: Dimension of the state vector.
        control_dim: Dimension of the control input.
        disturbance_dim: Dimension of the disturbance input (default 0).
    
    Methods:
        linearize: Linearizes the system around a given state, control, and disturbance.
        __call__: Evaluates the system dynamics.
    """
    dynamics_func: Callable
    state_dim: int
    control_dim: int
    disturbance_dim: int = 0

    def __init__(
        self,
        dynamics_func: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float], jnp.ndarray],
        state_dim: int,
        control_dim: int,
        disturbance_dim: int = 0,
    ) -> None:
        self.dynamics_func = dynamics_func
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim

    @eqx.filter_jit
    def linearize(
        self,
        state0: jnp.ndarray,
        control0: jnp.ndarray,
        disturbance0: Union[jnp.ndarray, None] = None,
        time: float = 0.0,
    ) -> jnp.ndarray:
        """
        Linearizes the system dynamics around a given operating point.
        This method computes the linear approximation of the system's dynamics at the specified state, control, and disturbance.
        It returns the result of applying the linearized system matrices to the provided inputs.
        Args:
            state0 (jnp.ndarray): The state vector at the operating point.
            control0 (jnp.ndarray): The control input vector at the operating point.
            disturbance0 (Union[jnp.ndarray, None], optional): The disturbance vector at the operating point.
                If None, a zero disturbance vector is used. Defaults to None.
            time (float, optional): The time at which to linearize the system. Defaults to 0.0.
        Returns:
            jnp.ndarray: The result of the linearized system, computed as A @ state0 + B @ control0 + C @ disturbance0 + D,
                where A, B, C, and D are the linearized system matrices.
        """
        if disturbance0 is None:
            disturbance0 = jnp.zeros((self.disturbance_dim,))
        A, B, C, D = linearize(self.dynamics_func, state0, control0, disturbance0, time)
        return A @ state0 + B @ control0 + C @ disturbance0 + D

    def __call__(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray,
        disturbance: Union[jnp.ndarray, None] = None,
        time: float = 0.0,
    ) -> jnp.ndarray:
        return self.dynamics_func(state, control, disturbance, time)


class ControlAffineDynamics(Dynamics):
    """
    Dynamics for control-affine systems: \dot{x} = f(x, t) + G(x, t)u
    
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
        self.drift_dynamics = drift_dynamics
        self.control_jacobian = control_jacobian

        def dynamics_func(
            x: jnp.ndarray,
            u: jnp.ndarray,
            d: Union[jnp.ndarray, None] = None,
            t: float = 0.0,
        ) -> jnp.ndarray:
            return drift_dynamics(x, t) + control_jacobian(x, t) @ u

        disturbance_dim = 0
        super().__init__(dynamics_func, state_dim, control_dim, disturbance_dim)

    @eqx.filter_jit
    def open_loop_dynamics(self, state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
        return self.drift_dynamics(state, time)


class ControlDisturbanceAffineDynamics(Dynamics):
    """
    Dynamics for control- and disturbance-affine systems:
        \dot{x} = f(x, t) + G(x, t)u + H(x, t)d
    
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
    ) -> None:
        self.drift_dynamics = drift_dynamics
        self.control_jacobian = control_jacobian
        self.disturbance_jacobian = disturbance_jacobian

        def dynamics_func(
            x: jnp.ndarray,
            u: jnp.ndarray,
            d: jnp.ndarray,
            t: float = 0.0,
        ) -> jnp.ndarray:
            return (
                drift_dynamics(x, t)
                + control_jacobian(x, t) @ u
                + disturbance_jacobian(x, t) @ d
            )

        super().__init__(dynamics_func, state_dim, control_dim, disturbance_dim)

    def open_loop_dynamics(self, state, time):
        return self.drift_dynamics(state, time)


class LinearControlDynamics(ControlAffineDynamics):
    """
    Linear time-invariant control system:
        \dot{x} = A x + B u + c
    
    Attributes:
        drift_matrix: The A matrix.
        control_matrix: The B matrix.
        constant: The c vector (default zero).
        disturbance_dim: Disturbance dimension (default 0).
    """
    drift_matrix: Union[jnp.ndarray]
    control_matrix: jnp.ndarray
    constant: Union[jnp.ndarray, None] = None
    disturbance_dim: int = 0

    def __init__(
        self,
        drift_matrix: jnp.ndarray,
        control_matrix: jnp.ndarray,
        constant: Union[jnp.ndarray, None] = None,
        disturbance_dim: int = 0,
    ) -> None:
        self.drift_matrix = drift_matrix
        self.control_matrix = control_matrix
        self.disturbance_dim = disturbance_dim
        state_dim: int = drift_matrix.shape[-1]
        control_dim: int = control_matrix.shape[-1]
        if constant is None:
            self.constant = jnp.zeros((state_dim,))
        else:
            assert constant.shape == (state_dim,)
            self.constant = constant

        def drift_dynamics(x: jnp.ndarray, t: float = 0.0) -> jnp.ndarray:
            return self.drift_matrix @ x + self.constant

        def control_jacobian_fn(x: jnp.ndarray, t: float = 0.0) -> jnp.ndarray:
            return self.control_matrix

        super().__init__(
            drift_dynamics,
            control_jacobian_fn,
            state_dim,
            control_dim,
        )


class LinearControlDisturbanceDynamics(ControlDisturbanceAffineDynamics):
    """
    Linear time-invariant control-disturbance system:
        \dot{x} = A x + B u + G d + c
    
    Attributes:
        drift_matrix: The A matrix.
        control_matrix: The B matrix.
        disturbance_matrix: The G matrix.
        constant: The c vector (default zero).
    """
    drift_matrix: jnp.ndarray
    control_matrix: jnp.ndarray
    disturbance_matrix: jnp.ndarray
    constant: Union[jnp.ndarray, None] = None

    def __init__(
        self,
        drift_matrix: jnp.ndarray,
        control_matrix: jnp.ndarray,
        disturbance_matrix: jnp.ndarray,
        constant: Union[jnp.ndarray, None] = None,
    ) -> None:
        self.drift_matrix = drift_matrix
        self.control_matrix = control_matrix
        self.disturbance_matrix = disturbance_matrix
        state_dim: int = drift_matrix.shape[-1]
        control_dim: int = control_matrix.shape[-1]
        disturbance_dim: int = disturbance_matrix.shape[-1]
        if constant is None:
            self.constant = jnp.zeros((state_dim,))
        else:
            assert constant.shape == (state_dim,)
            self.constant = constant

        def drift_dynamics(x: jnp.ndarray, t: float = 0.0) -> jnp.ndarray:
            return self.drift_matrix @ x + self.constant

        def control_jacobian_fn(x: jnp.ndarray, t: float = 0.0) -> jnp.ndarray:
            return self.control_matrix

        def disturbance_jacobian_fn(x: jnp.ndarray, t: float = 0.0) -> jnp.ndarray:
            return self.disturbance_matrix

        super().__init__(
            drift_dynamics,
            control_jacobian_fn,
            disturbance_jacobian_fn,
            state_dim,
            control_dim,
            disturbance_dim,
        )


def get_discrete_time_dynamics(
    continuous_time_dynamics: Dynamics, dt: float
) -> Dynamics:
    # Ensure the continuous_time_dynamics is an instance of Dynamics
    if not isinstance(continuous_time_dynamics, Dynamics):
        raise TypeError("continuous_time_dynamics must be an instance of Dynamics.")
    discete_dynamics: Callable[[jnp.ndarray, jnp.ndarray, Union[jnp.ndarray, None], float], jnp.ndarray] = runge_kutta_integrator(
        continuous_time_dynamics, dt
    )
    return Dynamics(
        discete_dynamics,
        continuous_time_dynamics.state_dim,
        continuous_time_dynamics.control_dim,
        continuous_time_dynamics.disturbance_dim,
    )


def get_linearized_dynamics_control_disturbance(
    dynamics: Dynamics,
    state0: jnp.ndarray,
    control0: jnp.ndarray,
    disturbance0: jnp.ndarray,
    time: float,
) -> LinearControlDisturbanceDynamics:
    """
    Linearizes a nonlinear dynamics function with respect to state, control, and disturbance at a given operating point.
    Args:
        dynamics (Dynamics): The nonlinear dynamics function to be linearized.
        state0 (jnp.ndarray): The state vector at the linearization point.
        control0 (jnp.ndarray): The control input vector at the linearization point.
        disturbance0 (jnp.ndarray): The disturbance vector at the linearization point.
        time (float): The time at which to linearize the dynamics.
    Returns:
        LinearControlDisturbanceDynamics: An object containing the linearized system matrices corresponding to the linearization of the dynamics around the specified point.
    """
    matrices: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] = linearize(
        dynamics, state0, control0, disturbance0, time
    )
    return LinearControlDisturbanceDynamics(*matrices)


def get_linearized_dynamics_control(
    dynamics: Dynamics,
    state0: jnp.ndarray,
    control0: jnp.ndarray,
    time: float,
) -> LinearControlDynamics:
    """
    Linearizes the given nonlinear dynamics around a specified state and control input.
    This function computes the linearized system matrices for the provided
    dynamics at the given state, control, and time, assuming zero disturbance. The
    resulting matrices are used to construct and return a LinearControlDynamics object.
    Args:
        dynamics (Dynamics): The nonlinear dynamics model to be linearized.
        state0 (jnp.ndarray): The state vector around which to linearize.
        control0 (jnp.ndarray): The control input vector around which to linearize.
        time (float): The time at which to linearize the dynamics.
    Returns:
        LinearControlDynamics: An object containing the linearized system matrices (A, B, D).
    """
    disturbance0 = jnp.zeros((dynamics.disturbance_dim,))
    matrices: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] = linearize(
        dynamics, state0, control0, disturbance0, time
    )
    A, B, _, D = matrices
    return LinearControlDynamics(A, B, D)
