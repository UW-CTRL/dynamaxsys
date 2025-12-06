import jax
import jax.numpy as jnp
import numpy as np
from dynamaxsys import (
    Dynamics,
    IntegratorND,
    TwoPlayerRelativeIntegratorND,
    SimpleCar,
    DynamicallyExtendedSimpleCar,
    RelativeSimpleCar,
    RelativeDynamicallyExtendedSimpleCar,
    Unicycle,
    DynamicallyExtendedUnicycle,
    RelativeUnicycle,
    RelativeDynamicallyExtendedUnicycle,
    get_discrete_time_dynamics,
    LinearControlDisturbanceDynamics,
    LinearControlDynamics,
    ControlDisturbanceAffineDynamics,
    ControlAffineDynamics,
    linearize,
    get_linearized_dynamics_control,
    get_linearized_dynamics_control_disturbance,
)


def test_LinearDynamics(with_disturbance=True):
    test_name = (
        "LinearControlDisturbanceDynamics"
        if with_disturbance
        else "LinearControlDynamics"
    )
    state_dim = 10
    control_dim = 2
    disturbance_dim = 1
    A = jax.random.normal(jax.random.PRNGKey(0), (state_dim, state_dim))
    B = jax.random.normal(jax.random.PRNGKey(1), (state_dim, control_dim))
    C = jax.random.normal(jax.random.PRNGKey(2), (state_dim, disturbance_dim))
    D = jax.random.normal(jax.random.PRNGKey(3), (state_dim,))
    dt = 0.1
    if with_disturbance:
        continuous_dynamics = LinearControlDisturbanceDynamics(A, B, C, D)
    else:
        continuous_dynamics = LinearControlDynamics(A, B, D)
    discrete_dynamics = get_discrete_time_dynamics(continuous_dynamics, dt=dt)
    state = jnp.ones((state_dim,))
    control = jnp.ones((control_dim,))
    disturbance = jnp.ones((disturbance_dim,))
    t = 0.0
    try:
        continuous_dynamics(state, control, disturbance, t)
    except TypeError:
        raise AssertionError("Error in continuous dynamics with disturbance input")

    try:
        discrete_dynamics(state, control, disturbance, t)
    except TypeError:
        raise AssertionError("Error in discrete dynamics with disturbance input")

    linear_matrices = linearize(continuous_dynamics, state, control, disturbance, t)

    assert jnp.isclose(
        jnp.array(linear_matrices[0]), continuous_dynamics.drift_matrix, atol=1e-6
    ).all(), (
        f"Expected {continuous_dynamics.drift_matrix}, got {jnp.array(linear_matrices[0])}"
    )

    assert jnp.isclose(
        jnp.array(linear_matrices[1]), continuous_dynamics.control_matrix, atol=1e-6
    ).all(), (
        f"Expected {continuous_dynamics.control_matrix}, got {jnp.array(linear_matrices[1])}"
    )

    if with_disturbance:
        assert jnp.isclose(
            jnp.array(linear_matrices[2]),
            continuous_dynamics.disturbance_matrix,
            atol=1e-6,
        ).all(), (
            f"Expected {continuous_dynamics.disturbance_matrix}, got {jnp.array(linear_matrices[2])}"
        )

    assert jnp.isclose(
        jnp.array(linear_matrices[3]), continuous_dynamics.constant, atol=1e-6
    ).all(), (
        f"Expected {continuous_dynamics.constant}, got {jnp.array(linear_matrices[3])}"
    )

    print(f"Passed: {test_name} class")


def test_AffineDynamics(with_disturbance=True):
    test_name = (
        "ControlDisturbanceAffineDynamics"
        if with_disturbance
        else "ControlAffineDynamics"
    )
    state_dim = 10
    control_dim = 2
    disturbance_dim = 1

    def drift_dynamics(x, t):
        return jnp.sin(x) + 0.1 * x

    def control_jacobian(x, t):
        return jnp.eye(state_dim, control_dim) * 0.5

    def disturbance_jacobian(x, t):
        return jnp.eye(state_dim, disturbance_dim) * 0.2

    dt = 0.1
    if with_disturbance:
        continuous_dynamics = ControlDisturbanceAffineDynamics(
            drift_dynamics,
            control_jacobian,
            disturbance_jacobian,
            state_dim,
            control_dim,
            disturbance_dim,
        )
    else:
        continuous_dynamics = ControlAffineDynamics(
            drift_dynamics,
            control_jacobian,
            state_dim,
            control_dim,
        )

    discrete_dynamics = get_discrete_time_dynamics(continuous_dynamics, dt=dt)
    state = jnp.ones((state_dim,))
    control = jnp.ones((control_dim,))
    disturbance = jnp.ones((disturbance_dim,))
    t = 0.0
    try:
        continuous_dynamics(state, control, disturbance, t)
    except TypeError:
        raise AssertionError("Error in continuous dynamics with disturbance input")

    try:
        discrete_dynamics(state, control, disturbance, t)
    except TypeError:
        raise AssertionError("Error in discrete dynamics with disturbance input")

    linear_matrices = linearize(continuous_dynamics, state, control, disturbance, t)

    assert jnp.isclose(
        jnp.array(linear_matrices[1]),
        continuous_dynamics.control_jacobian(state, t),
        atol=1e-6,
    ).all(), (
        f"Expected {continuous_dynamics.control_jacobian(state, t)}, got {jnp.array(linear_matrices[1])}"
    )

    if with_disturbance:
        assert jnp.isclose(
            jnp.array(linear_matrices[2]),
            continuous_dynamics.disturbance_jacobian(state, t),
            atol=1e-6,
        ).all(), (
            f"Expected {continuous_dynamics.disturbance_jacobian(state, t)}, got {jnp.array(linear_matrices[2])}"
        )

    print(f"Passed: {test_name} class")


def test_dynamics():
    # test on some made up dynamics
    def dynamics_func(
        state: jnp.ndarray, control: jnp.ndarray, disturbance: jnp.ndarray, time: float
    ) -> jnp.ndarray:
        return state**2 + control**2 + disturbance + time

    state_dim = 2
    control_dim = 2
    disturbance_dim = 2
    dt = 0.1

    ct_dynamics = Dynamics(
        dynamics_func=dynamics_func,
        state_dim=state_dim,
        control_dim=control_dim,
        disturbance_dim=disturbance_dim,
    )
    dt_dynamics = get_discrete_time_dynamics(ct_dynamics, dt)

    state = jnp.ones(state_dim)
    control = jnp.ones(control_dim)
    disturbance = jnp.ones(disturbance_dim)
    time = 0.0
    state_derivative = ct_dynamics(state, control, disturbance, time)
    xdot_linearize = ct_dynamics.linearize(state, control, disturbance, time)

    xnext = dt_dynamics(state, control, disturbance, time)
    xnext_linearize = dt_dynamics.linearize(state, control, disturbance, time)

    print("Passed: Dynamics class")


def test_continuous_time_unicycle():
    ct_dynamics = Unicycle()
    state = jnp.ones((ct_dynamics.state_dim,))
    control = jnp.ones((ct_dynamics.control_dim,))
    x, y, theta = state
    v, omega = control
    time = 0.0
    linear_dynamics = get_linearized_dynamics_control(ct_dynamics, state, control, time)

    analytic_A = jnp.array(
        [
            [0.0, 0.0, -v * jnp.sin(theta)],
            [0.0, 0.0, v * jnp.cos(theta)],
            [0.0, 0.0, 0.0],
        ]
    )
    analytic_B = jnp.array([[jnp.cos(theta), 0.0], [jnp.sin(theta), 0.0], [0.0, 1.0]])
    analytic_C = (
        jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), omega])
        - analytic_A @ state
        - analytic_B @ control
    )
    assert jnp.allclose(linear_dynamics.drift_matrix, analytic_A)
    assert jnp.allclose(linear_dynamics.control_matrix, analytic_B)
    assert jnp.allclose(linear_dynamics.constant, analytic_C)
    print("Passed: Unicycle")


def test_continuous_time_dynamic_unicycle():
    ct_dynamics = DynamicallyExtendedUnicycle()
    state = jnp.ones((ct_dynamics.state_dim,))
    control = jnp.ones((ct_dynamics.control_dim,))
    x, y, theta, v = state
    omega, a = control
    time = 0.0
    linear_dynamics = get_linearized_dynamics_control(ct_dynamics, state, control, time)

    analytic_A = jnp.array(
        [
            [0.0, 0.0, -v * jnp.sin(theta), jnp.cos(theta)],
            [0.0, 0.0, v * jnp.cos(theta), jnp.sin(theta)],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    analytic_B = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    analytic_C = (
        jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), omega, a])
        - analytic_A @ state
        - analytic_B @ control
    )
    assert jnp.allclose(linear_dynamics.drift_matrix, analytic_A)
    assert jnp.allclose(linear_dynamics.control_matrix, analytic_B)
    assert jnp.allclose(linear_dynamics.constant, analytic_C)
    print("Passed: Dynamically Extended Unicycle")


def test_continuous_time_simplecar():
    ct_dynamics = SimpleCar(wheelbase=1.0)
    state = jnp.ones((ct_dynamics.state_dim,))
    control = jnp.ones((ct_dynamics.control_dim,))
    x, y, theta = state
    v, tand = control
    time = 0.0
    linear_dynamics = get_linearized_dynamics_control(ct_dynamics, state, control, time)

    analytic_A = jnp.array(
        [
            [0.0, 0.0, -v * jnp.sin(theta)],
            [0.0, 0.0, v * jnp.cos(theta)],
            [0.0, 0.0, 0.0],
        ]
    )
    analytic_B = jnp.array(
        [
            [jnp.cos(theta), 0.0],
            [jnp.sin(theta), 0.0],
            [tand / ct_dynamics.wheelbase, v / ct_dynamics.wheelbase],
        ]
    )
    analytic_C = (
        jnp.array(
            [v * jnp.cos(theta), v * jnp.sin(theta), v / ct_dynamics.wheelbase * tand]
        )
        - analytic_A @ state
        - analytic_B @ control
    )
    assert jnp.allclose(linear_dynamics.drift_matrix, analytic_A)
    assert jnp.allclose(linear_dynamics.control_matrix, analytic_B)
    assert jnp.allclose(linear_dynamics.constant, analytic_C)
    print("Passed: Simple Car")


def test_continuous_time_dynamic_simplecar():
    ct_dynamics = DynamicallyExtendedSimpleCar(wheelbase=1.0)
    state = jnp.ones((ct_dynamics.state_dim,))
    control = jnp.ones((ct_dynamics.control_dim,))
    x, y, theta, v = state
    tand, a = control
    time = 0.0
    linear_dynamics = get_linearized_dynamics_control(ct_dynamics, state, control, time)

    analytic_A = jnp.array(
        [
            [0.0, 0.0, -v * jnp.sin(theta), jnp.cos(theta)],
            [0.0, 0.0, v * jnp.cos(theta), jnp.sin(theta)],
            [0.0, 0.0, 0.0, tand / ct_dynamics.wheelbase],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    analytic_B = jnp.array(
        [[0.0, 0.0], [0.0, 0.0], [v / ct_dynamics.wheelbase, 0.0], [0.0, 1.0]]
    )
    analytic_C = (
        jnp.array(
            [
                v * jnp.cos(theta),
                v * jnp.sin(theta),
                v / ct_dynamics.wheelbase * tand,
                a,
            ]
        )
        - analytic_A @ state
        - analytic_B @ control
    )
    assert jnp.allclose(linear_dynamics.drift_matrix, analytic_A)
    assert jnp.allclose(linear_dynamics.control_matrix, analytic_B)
    assert jnp.allclose(linear_dynamics.constant, analytic_C)
    print("Passed: Dynamically Extended Simple Car")


if __name__ == "__main__":
    test_dynamics()
    test_AffineDynamics(with_disturbance=True)
    test_AffineDynamics(with_disturbance=False)
    test_LinearDynamics(with_disturbance=True)
    test_LinearDynamics(with_disturbance=False)

    test_continuous_time_unicycle()
    test_continuous_time_dynamic_unicycle()

    test_continuous_time_simplecar()
    test_continuous_time_dynamic_simplecar()
