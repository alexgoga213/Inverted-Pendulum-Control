"""Nonlinear cart-pole model derived with Lagrangian mechanics.

State  x = [p, p_dot, theta, theta_dot]
    p      cart position [m]
    theta  pole angle from the UPRIGHT position [rad], positive when the pole leans towards +p
Input  u = horizontal force on the cart [N]

The pole is a uniform rod of mass m and half-length l (pivot -> centre of mass), so
I_cm = m l^2 / 3 and J = I_cm + m l^2 is its inertia about the pivot.

Lagrangian:
    T = 1/2 (M + m) p_dot^2 + m l cos(theta) p_dot theta_dot + 1/2 J theta_dot^2
    V = m g l cos(theta)
Euler-Lagrange equations (with viscous cart friction b and pivot friction d):
    (M + m) p_ddot + m l cos(theta) theta_ddot - m l sin(theta) theta_dot^2 = u - b p_dot
    m l cos(theta) p_ddot + J theta_ddot - m g l sin(theta)                   = -d theta_dot
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

STATE_LABELS = ("cart position [m]", "cart velocity [m/s]", "pole angle [rad]", "angular velocity [rad/s]")


@dataclass(frozen=True)
class CartPoleParams:
    M: float = 1.0  # cart mass [kg]
    m: float = 0.2  # pole mass [kg]
    l: float = 0.3  # half pole length [m]
    b: float = 0.1  # cart viscous friction [N s/m]
    d: float = 0.002  # pivot viscous friction [N m s/rad]
    g: float = 9.81  # gravity [m/s^2]
    u_max: float = 20.0  # actuator saturation [N]
    track_limit: float = 2.0  # |p| must stay below this [m]

    @property
    def J(self) -> float:
        """Pole inertia about the pivot."""
        return self.m * self.l**2 / 3.0 + self.m * self.l**2

    def perturbed(self, **scale: float) -> "CartPoleParams":
        """Copy with some parameters multiplied, e.g. perturbed(m=1.5) for +50% pole mass."""
        return replace(self, **{k: getattr(self, k) * v for k, v in scale.items()})


def wrap_angle(theta: float | np.ndarray) -> float | np.ndarray:
    """Map an angle to [-pi, pi)."""
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def dynamics(x: np.ndarray, u: float, p: CartPoleParams) -> np.ndarray:
    """Continuous-time nonlinear dynamics x_dot = f(x, u)."""
    _, pos_dot, th, th_dot = x
    s, c = np.sin(th), np.cos(th)
    ml = p.m * p.l

    # Mass matrix * [p_ddot, theta_ddot] = rhs
    a11, a12 = p.M + p.m, ml * c
    a21, a22 = ml * c, p.J
    r1 = u - p.b * pos_dot + ml * s * th_dot**2
    r2 = ml * p.g * s - p.d * th_dot

    det = a11 * a22 - a12 * a21  # > 0 for every theta
    p_ddot = (a22 * r1 - a12 * r2) / det
    th_ddot = (a11 * r2 - a21 * r1) / det
    return np.array([pos_dot, p_ddot, th_dot, th_ddot])


def rk4_step(x: np.ndarray, u: float, dt: float, p: CartPoleParams) -> np.ndarray:
    """One 4th-order Runge-Kutta step with the input held constant (zero-order hold)."""
    k1 = dynamics(x, u, p)
    k2 = dynamics(x + 0.5 * dt * k1, u, p)
    k3 = dynamics(x + 0.5 * dt * k2, u, p)
    k4 = dynamics(x + dt * k3, u, p)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def linearize(p: CartPoleParams) -> tuple[np.ndarray, np.ndarray]:
    """Analytic linearization around the upright equilibrium x = 0, u = 0.

    Returns (A, B) such that x_dot ~= A x + B u.
    """
    ml = p.m * p.l
    det = (p.M + p.m) * p.J - ml**2
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -p.J * p.b / det, -ml**2 * p.g / det, ml * p.d / det],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, ml * p.b / det, (p.M + p.m) * ml * p.g / det, -(p.M + p.m) * p.d / det],
        ]
    )
    B = np.array([[0.0], [p.J / det], [0.0], [-ml / det]])
    return A, B


def discretize(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretization via the matrix exponential of [[A, B], [0, 0]]."""
    from scipy.linalg import expm

    n, m = B.shape
    big = np.zeros((n + m, n + m))
    big[:n, :n], big[:n, n:] = A, B
    phi = expm(big * dt)
    return phi[:n, :n], phi[:n, n:]


def energy(x: np.ndarray, p: CartPoleParams) -> float:
    """Pole energy relative to the upright position (0 when balanced, -2 m g l hanging down)."""
    th, th_dot = x[2], x[3]
    return 0.5 * p.J * th_dot**2 + p.m * p.g * p.l * (np.cos(th) - 1.0)
