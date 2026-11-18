"""Linear Model Predictive Control with hard constraints.

At every sample the controller solves a quadratic program over a horizon of N steps:

    min   sum_{k=0}^{N-1} (x_k - r)^T Q (x_k - r) + u_k^T R u_k  +  (x_N - r)^T P (x_N - r)
    s.t.  x_{k+1} = A x_k + B u_k          (linearized, discretized model)
          |u_k|   <= u_max                 (actuator limit)
          |p_k|   <= track_limit - margin  (the cart must stay on the rail)
          x_0     =  measured state

and applies only the first input u_0 (receding horizon). The terminal cost P is the LQR
Riccati solution, so the MPC behaves like LQR when no constraint is active, but plans
ahead and respects limits that LQR simply ignores.

Implementation notes
* The problem is written in error coordinates e = x - r. For the references used here (a cart
  position with zero velocity and an upright pole) r is an equilibrium of the model, A r = r, so
  the error obeys the same dynamics e_{k+1} = A e_k + B u_k.
* Because of that the problem is DPP (Disciplined Parametrized Programming): CVXPY compiles it
  once, and every sample only updates the parameters (e_0 and the target position) before
  re-solving with OSQP, warm-started from the previous solution.
"""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

from pendulum.controllers.base import Controller
from pendulum.controllers.lqr import DEFAULT_Q, DEFAULT_R, dlqr
from pendulum.model import CartPoleParams, discretize, linearize, wrap_angle


class MPC(Controller):
    name = "MPC"

    def __init__(
        self,
        dt: float,
        params: CartPoleParams,
        horizon: int = 30,
        Q: np.ndarray = DEFAULT_Q,
        R: np.ndarray = DEFAULT_R,
        track_margin: float = 0.1,
    ) -> None:
        super().__init__(dt)
        A, B = linearize(params)
        Ad, Bd = discretize(A, B, dt)
        K_lqr, P = dlqr(Ad, Bd, Q, R)
        self._fallback_K = K_lqr
        self.horizon = horizon

        n, m = Bd.shape
        self._e0 = cp.Parameter(n)  # initial error x0 - r
        self._p_ref = cp.Parameter()  # target cart position (for the track constraint)
        E = cp.Variable((n, horizon + 1))
        U = cp.Variable((m, horizon))

        cost = 0
        cons = [E[:, 0] == self._e0]
        p_lim = params.track_limit - track_margin
        for k in range(horizon):
            cost += cp.quad_form(E[:, k], Q) + cp.quad_form(U[:, k], R)
            cons += [
                E[:, k + 1] == Ad @ E[:, k] + Bd @ U[:, k],
                cp.abs(U[:, k]) <= params.u_max,
                cp.abs(E[0, k + 1] + self._p_ref) <= p_lim,  # absolute cart position
            ]
        cost += cp.quad_form(E[:, horizon], P)

        self._U = U
        self._problem = cp.Problem(cp.Minimize(cost), cons)
        assert self._problem.is_dpp()
        self.solve_times: list[float] = []
        self.fallbacks = 0

    def reset(self) -> None:
        self.solve_times = []
        self.fallbacks = 0

    def compute(self, x: np.ndarray, x_ref: np.ndarray) -> float:
        err = np.asarray(x, dtype=float) - x_ref
        err[2] = wrap_angle(err[2])
        self._e0.value = err
        self._p_ref.value = float(x_ref[0])

        t0 = time.perf_counter()
        try:
            self._problem.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-4, eps_rel=1e-4, max_iter=10000)
            status = self._problem.status
        except cp.error.SolverError:
            status = "solver_error"
        self.solve_times.append(time.perf_counter() - t0)

        if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and self._U.value is not None:
            return float(self._U.value[0, 0])
        # Infeasible (e.g. state already outside the constraints): fall back to LQR.
        self.fallbacks += 1
        return float(-(self._fallback_K @ err)[0])
