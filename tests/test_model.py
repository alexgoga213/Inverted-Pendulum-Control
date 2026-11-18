from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import cont2discrete

from pendulum.model import CartPoleParams, discretize, dynamics, linearize, rk4_step, wrap_angle

P = CartPoleParams()


def total_energy(x, p):
    """Cart + pole mechanical energy (absolute), used to check the integrator."""
    pos_dot, th, th_dot = x[1], x[2], x[3]
    T = 0.5 * (p.M + p.m) * pos_dot**2 + p.m * p.l * np.cos(th) * pos_dot * th_dot + 0.5 * p.J * th_dot**2
    V = p.m * p.g * p.l * np.cos(th)
    return T + V


def test_upright_is_an_equilibrium():
    np.testing.assert_allclose(dynamics(np.zeros(4), 0.0, P), 0.0, atol=1e-12)


def test_hanging_down_is_an_equilibrium():
    np.testing.assert_allclose(dynamics(np.array([0, 0, np.pi, 0]), 0.0, P), 0.0, atol=1e-12)


def test_gravity_tips_the_pole_further():
    # Leaning towards +p, the pole must accelerate further towards +p (upright is unstable).
    assert dynamics(np.array([0, 0, 0.1, 0]), 0.0, P)[3] > 0


def test_linearization_matches_finite_differences():
    A, B = linearize(P)
    eps = 1e-6
    A_fd = np.zeros((4, 4))
    for i in range(4):
        dx = np.zeros(4)
        dx[i] = eps
        A_fd[:, i] = (dynamics(dx, 0.0, P) - dynamics(-dx, 0.0, P)) / (2 * eps)
    B_fd = ((dynamics(np.zeros(4), eps, P) - dynamics(np.zeros(4), -eps, P)) / (2 * eps)).reshape(4, 1)
    np.testing.assert_allclose(A, A_fd, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(B, B_fd, rtol=1e-6, atol=1e-8)


def test_upright_linear_model_is_unstable_and_controllable():
    A, B = linearize(P)
    assert np.max(np.real(np.linalg.eigvals(A))) > 0
    ctrb = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(4)])
    assert np.linalg.matrix_rank(ctrb) == 4


def test_discretization_matches_scipy():
    A, B = linearize(P)
    Ad, Bd = discretize(A, B, 0.02)
    Ad_ref, Bd_ref, *_ = cont2discrete((A, B, np.eye(4), np.zeros((4, 1))), 0.02, method="zoh")
    np.testing.assert_allclose(Ad, Ad_ref, atol=1e-12)
    np.testing.assert_allclose(Bd, Bd_ref, atol=1e-12)


def test_rk4_conserves_energy_without_friction():
    p = replace(P, b=0.0, d=0.0)
    x = np.array([0.0, 0.3, 2.5, 0.0])  # large swing, cart moving
    e0 = total_energy(x, p)
    for _ in range(3000):  # 3 s
        x = rk4_step(x, 0.0, 0.001, p)
    assert abs(total_energy(x, p) - e0) < 1e-7


@pytest.mark.parametrize("theta, expected", [(0.0, 0.0), (2 * np.pi, 0.0), (np.pi + 0.1, -np.pi + 0.1), (-0.5, -0.5)])
def test_wrap_angle(theta, expected):
    assert wrap_angle(theta) == pytest.approx(expected)


def test_perturbed_parameters():
    q = P.perturbed(m=1.5, l=2.0)
    assert q.m == pytest.approx(0.3) and q.l == pytest.approx(0.6) and q.M == P.M
