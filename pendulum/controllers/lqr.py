"""Discrete-time Linear Quadratic Regulator.

Minimises J = sum_k (x_k - x_ref)^T Q (x_k - x_ref) + u_k^T R u_k for the linearized, ZOH-discretized
model. The optimal gain comes from the Discrete Algebraic Riccati Equation (DARE):

    P = A^T P A - A^T P B (R + B^T P B)^-1 B^T P A + Q
    K = (R + B^T P B)^-1 B^T P A,     u = -K (x - x_ref)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

from pendulum.controllers.base import Controller
from pendulum.model import CartPoleParams, discretize, linearize, wrap_angle

DEFAULT_Q = np.diag([20.0, 1.0, 100.0, 1.0])
DEFAULT_R = np.array([[0.05]])


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


class LQR(Controller):
    name = "LQR"

    def __init__(
        self,
        dt: float,
        params: CartPoleParams,
        Q: np.ndarray = DEFAULT_Q,
        R: np.ndarray = DEFAULT_R,
    ) -> None:
        super().__init__(dt)
        A, B = linearize(params)
        self.Ad, self.Bd = discretize(A, B, dt)
        self.Q, self.R = Q, R
        self.K, self.P = dlqr(self.Ad, self.Bd, Q, R)

    @property
    def closed_loop_poles(self) -> np.ndarray:
        """Eigenvalues of A - B K (all strictly inside the unit circle for a stable loop)."""
        return np.linalg.eigvals(self.Ad - self.Bd @ self.K)

    def compute(self, x: np.ndarray, x_ref: np.ndarray) -> float:
        err = np.asarray(x, dtype=float) - x_ref
        err[2] = wrap_angle(err[2])  # so that theta = 2*pi is treated as upright
        return float(-(self.K @ err)[0])
