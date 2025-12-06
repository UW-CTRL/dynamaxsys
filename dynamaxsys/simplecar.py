import jax.numpy as jnp
from dynamaxsys.base import (
    ControlAffineDynamics,
    Dynamics,
    ControlDisturbanceAffineDynamics,
)
from typing import Union


class SimpleCar(Dynamics):
    state_dim: int = 3
    control_dim: int = 2
    wheelbase: float
    """ Simple car model with state [x, y, theta] and control [v, tandelta]
    where x,y is the position, theta is the heading angle,
    v is the linear velocity, and tandelta is the tangent of the steering angle.
    The dynamics are given by:
        dx/dt = v * cos(theta)
        dy/dt = v * sin(theta)
        dtheta/dt = v / L * tandelta
    where L is the wheelbase of the car.
    """

    def __init__(self, wheelbase: float) -> None:
        self.wheelbase = wheelbase

        def dynamics_func(
            state: jnp.ndarray,
            control: jnp.ndarray,
            disturbance: Union[jnp.ndarray, None] = None,
            time: float = 0.0,
        ) -> jnp.ndarray:
            x, y, th = state
            v, tandelta = control
            return jnp.array(
                [v * jnp.cos(th), v * jnp.sin(th), v / self.wheelbase * tandelta]
            )

        super().__init__(dynamics_func, self.state_dim, self.control_dim)


class DynamicallyExtendedSimpleCar(ControlAffineDynamics):
    state_dim: int = 4
    control_dim: int = 2
    wheelbase: float
    min_max_velocity: tuple
    """ Dynamically extended simple car model with state [x, y, theta, v] and control [tandelta, a]
    where x,y is the position, theta is the heading angle,
    v is the linear velocity, tandelta is the tangent of the steering angle,
    and a is the linear acceleration.
    The dynamics are given by:
        dx/dt = v * cos(theta)
        dy/dt = v * sin(theta)
        dtheta/dt = v / L * tandelta
        dv/dt = a
    where L is the wheelbase of the car.
    """

    def __init__(
        self,
        wheelbase: float,
        min_max_velocity: tuple = (-jnp.inf, jnp.inf),
    ) -> None:
        self.wheelbase = wheelbase
        self.min_max_velocity = min_max_velocity

        def drift_dynamics(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            x, y, th, v = state
            v = jnp.clip(v, *self.min_max_velocity)
            # tandelta, a = control
            return jnp.array(
                [
                    v * jnp.cos(th),
                    v * jnp.sin(th),
                    0.0,
                    0.0,
                ]
            )

        def control_jacobian(state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
            # tandelta, a = control, tandelta = tan(delta)
            x, y, th, v = state
            v = jnp.clip(v, *self.min_max_velocity)

            return jnp.array(
                [[0.0, 0.0], [0.0, 0.0], [v / self.wheelbase, 0.0], [0.0, 1.0]]
            )

        super().__init__(
            drift_dynamics, control_jacobian, self.state_dim, self.control_dim
        )


class RelativeSimpleCar(Dynamics):
    state_dim: int = 3
    control_dim: int = 2
    disturbance_dim: int = 2
    wheelbase_ego: float
    wheelbase_contender: float
    """ Relative simple car model with state [xR, yR, threl] and control [v1, tandelta1] and disturbance [v2, tandelta2]
    where xR, yR is the position of the contender relative to the ego car,
    threl is the heading of the contender relative to the ego car,
    v1, tandelta1 are the linear velocity and tangent of the steering angle of the ego car,
    and v2, tandelta2 are the linear velocity and tangent of the steering angle of the contender car.
    The dynamics are given by:
        dxR/dt = v2 * cos(threl) - v1 + yR * v1 / L * tandelta1
        dyR/dt = v2 * sin(threl) - xR * v1 / L * tandelta1
        dthrel/dt = v2 / L * tandelta2 - v1 / L * tandelta1
    where L is the wheelbase of the car."""

    def __init__(self, wheelbase_ego: float, wheelbase_contender: float) -> None:
        self.wheelbase_ego = wheelbase_ego
        self.wheelbase_contender = wheelbase_contender

        def dynamics_func(
            state: jnp.ndarray,
            control: jnp.ndarray,
            disturbance: jnp.ndarray,
            time: float = 0.0,
        ) -> jnp.ndarray:
            xR, yR, threl = state
            v1, tandelta1 = control
            v2, tandelta2 = disturbance
            return jnp.array(
                [
                    v2 * jnp.cos(threl) - v1 + yR * v1 / self.wheelbase_ego * tandelta1,
                    v2 * jnp.sin(threl) - xR * v1 / self.wheelbase_ego * tandelta1,
                    v2 / self.wheelbase_contender * tandelta2 - v1 / self.wheelbase_ego * tandelta1,
                ]
            )

        super().__init__(
            dynamics_func, self.state_dim, self.control_dim, self.disturbance_dim
        )


class RelativeDynamicallyExtendedSimpleCar(ControlDisturbanceAffineDynamics):
    state_dim: int = 5
    control_dim: int = 2
    disturbance_dim: int = 2
    min_max_velocity_ego: tuple
    min_max_velocity_contender: tuple
    wheelbase_ego: float
    wheelbase_contender: float
    """ Relative dynamically extended simple car model with state [xR, yR, threl, v1, v2],
    control [tandelta1, a1] and disturbance [tandelta2, a2],
    where xR, yR is the position of the contender relative to the ego car,
    threl is the heading of the contender relative to the ego car,
    v1, a1 are the linear velocity and acceleration of the ego car,
    and v2, a2 are the linear velocity and acceleration of the contender car.
    The dynamics are given by:
        dxR/dt = v2 * cos(threl) - v1 + yR * v1 / L1 * tandelta1
        dyR/dt = v2 * sin(threl) - xR * v1 / L1 * tandelta1
        dthrel/dt = v2 / L2 * tandelta2 - v1 / L1 * tandelta1
        dv1/dt = a1
        dv2/dt = a2
    where L1 is the wheelbase of the ego car and L2 is the wheelbase of the contender car.
    """

    def __init__(
        self,
        wheelbase_ego: float,
        wheelbase_contender: float,
        min_max_velocity_ego: tuple = (-jnp.inf, jnp.inf),
        min_max_velocity_contender: tuple = (-jnp.inf, jnp.inf),
    ):
        self.wheelbase_ego = wheelbase_ego
        self.wheelbase_contender = wheelbase_contender
        self.min_max_velocity_ego = min_max_velocity_ego
        self.min_max_velocity_contender = min_max_velocity_contender

        def drift_dynamics(state: jnp.ndarray, time: float) -> jnp.ndarray:
            xrel, yrel, threl, v1, v2 = state
            # om1, a1, om2, a1 = control
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

        def control_jacobian(state: jnp.ndarray, time: float) -> jnp.ndarray:
            xrel, yrel, threl, v1, v2 = state
            # om1, a1, om2, a1 = control
            v1 = jnp.clip(v1, *self.min_max_velocity_ego)
            return jnp.array(
                [
                    [yrel * v1 / self.wheelbase_ego, 0.0],
                    [-xrel * v1 / self.wheelbase_ego, 0.0],
                    [-v1 / self.wheelbase_ego, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            )

        def disturbance_jacobian(state: jnp.ndarray, time: float) -> jnp.ndarray:
            xrel, yrel, threl, v1, v2 = state
            # om2, a2 = disturbance
            v2 = jnp.clip(v2, *self.min_max_velocity_contender)
            return jnp.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [v2 / self.wheelbase_contender, 0.0],
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
