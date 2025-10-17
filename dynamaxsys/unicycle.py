import jax.numpy as jnp
from dynamaxsys.base import ControlAffineDynamics, ControlDisturbanceAffineDynamics


class Unicycle(ControlAffineDynamics):
    state_dim: int = 3
    control_dim: int = 2

    def __init__(self):
        def drift_dynamics(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            return jnp.array([0.0, 0.0, 0.0])

        def control_jacobian(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            _, _, th = state
            # v, om = control
            return jnp.array([[jnp.cos(th), 0.0], [jnp.sin(th), 0.0], [0.0, 1.0]])

        super().__init__(
            drift_dynamics, control_jacobian, self.state_dim, self.control_dim
        )


class DynamicallyExtendedUnicycle(ControlAffineDynamics):
    state_dim: int = 4
    control_dim: int = 2

    def __init__(self):
        def drift_dynamics(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            x, y, th, v = state
            # v, om = control
            return jnp.array([v * jnp.cos(th), v * jnp.sin(th), 0.0, 0.0])

        def control_jacobian(state, time: float = 0.0) -> jnp.ndarray:
            x, y, th, v = state
            # v, om = control
            return jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        super().__init__(
            drift_dynamics, control_jacobian, self.state_dim, self.control_dim
        )


class RelativeUnicycle(ControlDisturbanceAffineDynamics):
    state_dim: int = 3
    control_dim: int = 2
    disturbance_dim: int = 2

    # def ode_dynamics(self, state, control, time=0):
    #     xrel, yrel, threl = state
    #     v1, om1, v2, om2 = control
    #     return jnp.array([v2 * jnp.cos(threl) + om1 * yrel - v1,
    #                       v2 * jnp.sin(threl) - om1 * xrel,
    #                       om2 - om1])

    def __init__(self):
        def drift_dynamics(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            xrel, yrel, threl = state
            # v1, om1, v2, om2 = control
            return jnp.zeros(self.state_dim)

        def control_jacobian(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            xrel, yrel, threl = state
            # v1, om1 = control
            return jnp.array(
                [
                    [-1.0, yrel],
                    [0.0, -xrel],
                    [0.0, -1.0],
                ]
            )

        def disturbance_jacobian(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            xrel, yrel, threl = state
            # v2, om2 = disturbance
            return jnp.array(
                [
                    [jnp.cos(threl), 0.0],
                    [jnp.sin(threl), 0.0],
                    [0.0, 1.0],
                ]
            )

        super().__init__(
            drift_dynamics,
            control_jacobian,
            disturbance_jacobian,
            self.state_dim,
            self.control_dim,
            self.disturbance_dim,
        )


class RelativeDynamicallyExtendedUnicycle(ControlDisturbanceAffineDynamics):
    state_dim: int = 5
    control_dim: int = 2
    disturbance_dim: int = 2

    def __init__(self):
        def drift_dynamics(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            xrel, yrel, threl, v1, v2 = state
            # om1, a1, om2, a1 = control
            return jnp.array(
                [
                    v2 * jnp.cos(threl) - v1,
                    v2 * jnp.sin(threl),
                    0.0,
                    0.0,
                    0.0,
                ]
            )

        def control_jacobian(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            xrel, yrel, threl, v1, v2 = state
            # om1, a1, om2, a1 = control
            return jnp.array(
                [
                    [yrel, 0.0],
                    [xrel, 0.0],
                    [-1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            )

        def disturbance_jacobian(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            xrel, yrel, threl, v1, v2 = state
            # om2, a2 = disturbance
            return jnp.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                ]
            )

        super().__init__(
            drift_dynamics,
            control_jacobian,
            disturbance_jacobian,
            self.state_dim,
            self.control_dim,
            self.disturbance_dim,
        )
