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

    def __init__(
        self,
        wheelbase: float,
        min_max_velocity: tuple = (0.0, 5.0),
    ) -> None:
        self.wheelbase = wheelbase
        self.min_max_velocity = min_max_velocity

        def drift_dynamics(state: jnp.ndarray, time: float = 0) -> jnp.ndarray:
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

        def control_jacobian(state: jnp.ndarray, time: float = 0) -> jnp.ndarray:
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
    wheelbase: float

    def __init__(self, wheelbase: float) -> None:
        self.wheelbase = wheelbase

        def dynamics_func(
            state: jnp.ndarray,
            control: jnp.ndarray,
            disturbance: jnp.ndarray,
            time: float = 0,
        ) -> jnp.ndarray:
            xR, yR, threl = state
            v1, tandelta1 = control
            v2, tandelta2 = disturbance
            return jnp.array(
                [
                    v2 * jnp.cos(threl) - v1 + yR * v1 / self.wheelbase * tandelta1,
                    v2 * jnp.sin(threl),
                    -xR * v1 / self.wheelbase * tandelta1,
                    v2 / self.wheelbase * tandelta2 - v1 / self.wheelbase * tandelta1,
                ]
            )

        super().__init__(dynamics_func, self.state_dim, self.control_dim, self.disturbance_dim)


class RelativeDynamicallyExtendedSimpleCar(ControlDisturbanceAffineDynamics):
    state_dim: int = 5
    control_dim: int = 2
    disturbance_dim: int = 2
    min_max_velocity: tuple
    wheelbase_ego: float
    wheelbase_contender: float


    def __init__(
        self,
        wheelbase_ego: float,
        wheelbase_contender: float,
        min_max_velocity: tuple = (0.0, 5.0),
    ):
        self.wheelbase_ego = wheelbase_ego
        self.wheelbase_contender = wheelbase_contender
        self.min_max_velocity = min_max_velocity

        def drift_dynamics(state: jnp.ndarray, time: float) -> jnp.ndarray:
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

        def control_jacobian(state: jnp.ndarray, time: float) -> jnp.ndarray:
            xrel, yrel, threl, v1, v2 = state
            # om1, a1, om2, a1 = control
            return jnp.array(
                [
                    [yrel * v1 / self.wheelbase_ego, 0.0],
                    [xrel * v1 / self.wheelbase_ego, 0.0],
                    [-v1 / self.wheelbase_ego, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            )

        def disturbance_jacobian(state: jnp.ndarray, time: float) -> jnp.ndarray:
            xrel, yrel, threl, v1, v2 = state
            # om2, a2 = disturbance
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
