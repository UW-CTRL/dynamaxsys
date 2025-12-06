import jax.numpy as jnp
from dynamaxsys.base import ControlAffineDynamics, ControlDisturbanceAffineDynamics


class Unicycle(ControlAffineDynamics):
    state_dim: int = 3
    control_dim: int = 2
    """ Unicycle model with state [x, y, theta] and control [v, omega]
    where x,y is the position, theta is the heading angle,
    v is the linear velocity, and omega is the angular velocity.
    The dynamics are given by:
        dx/dt = v * cos(theta)
        dy/dt = v * sin(theta)
        dtheta/dt = omega
    """

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
    min_max_velocity: tuple
    """ Dynamically extended unicycle model with state [x, y, theta, v] and control [a, omega]
    where x,y is the position, theta is the heading angle,
    v is the linear velocity, a is the linear acceleration, and omega is the angular velocity.
    The dynamics are given by:
        dx/dt = v * cos(theta)
        dy/dt = v * sin(theta)
        dtheta/dt = omega
        dv/dt = a
    """

    def __init__(self, min_max_velocity: tuple = (-jnp.inf, jnp.inf)):
        self.min_max_velocity = min_max_velocity

        def drift_dynamics(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            _, _, th, v = state
            v = jnp.clip(v, *self.min_max_velocity)
            return jnp.array([v * jnp.cos(th), v * jnp.sin(th), 0.0, 0.0])

        def control_jacobian(state, time: float = 0.0) -> jnp.ndarray:
            return jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        super().__init__(
            drift_dynamics, control_jacobian, self.state_dim, self.control_dim
        )


class RelativeUnicycle(ControlDisturbanceAffineDynamics):
    state_dim: int = 3
    control_dim: int = 2
    disturbance_dim: int = 2
    """ Relative unicycle dynamics with state [x_rel, y_rel, theta_rel],
    control [v1, omega1] and disturbance [v2, omega2],
    where (x_rel, y_rel) is the position of unicycle 2 relative to unicycle 1,
    theta_rel is the heading of unicycle 2 relative to unicycle 1,
    v1, omega1 are the linear and angular velocities of unicycle 1,
    and v2, omega2 are the linear and angular velocities of unicycle 2.
    The dynamics are given by:
        dx_rel/dt = v2 * cos(theta_rel) + omega1 * y_rel - v1
        dy_rel/dt = v2 * sin(theta_rel) - omega1 * x_rel
        dtheta_rel/dt = omega2 - omega1
    """

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
    min_max_velocity_ego: tuple
    min_max_velocity_contender: tuple
    """ Relative dynamically extended unicycle dynamics with state [x_rel, y_rel, theta_rel, v1, v2],
    control [a1, omega1] and disturbance [a2, omega2],
    where (x_rel, y_rel) is the position of unicycle 2 relative to unicycle 1,
    theta_rel is the heading of unicycle 2 relative to unicycle 1,
    v1, a1 are the linear velocity and acceleration of unicycle 1,
    and v2, a2 are the linear velocity and acceleration of unicycle 2.
    The dynamics are given by:
        dx_rel/dt = v2 * cos(theta_rel) + omega1 * y_rel - v1
        dy_rel/dt = v2 * sin(theta_rel) - omega1 * x_rel
        dtheta_rel/dt = omega2 - omega1
        dv1/dt = a1
        dv2/dt = a2
    """

    def __init__(
        self,
        min_max_velocity_ego: tuple = (-jnp.inf, jnp.inf),
        min_max_velocity_contender: tuple = (-jnp.inf, jnp.inf),
    ):
        self.min_max_velocity_ego = min_max_velocity_ego
        self.min_max_velocity_contender = min_max_velocity_contender

        def drift_dynamics(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            xrel, yrel, threl, v1, v2 = state
            v1 = jnp.clip(v1, *self.min_max_velocity_ego)
            v2 = jnp.clip(v2, *self.min_max_velocity_contender)
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
            xrel, yrel, _, _, _ = state
            # om1, a1 = control
            return jnp.array(
                [
                    [yrel, 0.0],
                    [-xrel, 0.0],
                    [-1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            )

        def disturbance_jacobian(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            # xrel, yrel, threl, v1, v2 = state
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
