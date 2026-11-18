"""Energy-based swing-up with automatic hand-over to LQR (hybrid controller).

1. Swing-up (Astrom & Furuta, 2000). The pole energy relative to upright,
       E = 1/2 J theta_dot^2 + m g l (cos(theta) - 1),
   is 0 at the top and -2 m g l hanging down. Moving the pivot with acceleration a changes it at
       dE/dt = -m l a theta_dot cos(theta),
   so the cart acceleration
       a = k_E * E * sign(theta_dot cos(theta))
   always pumps energy towards E = 0 (as long as E < 0). A weak PD term keeps the cart centred.
   The desired acceleration is turned into a force with the exact inverse dynamics of the model.

2. Catch. When the pole is close to upright the controller switches to LQR. If the pole falls
   again (|theta| too large) it switches back to swing-up: a simple hybrid automaton with
   hysteresis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pendulum.controllers.base import Controller
from pendulum.controllers.lqr import LQR
from pendulum.model import CartPoleParams, energy, wrap_angle


@dataclass
class SwingUpGains:
    # Tuned with a grid search over 15 different starting states (all 15 succeed, the cart stays
    # within ~0.7 m of the centre). See tests/test_controllers.py.
    k_energy: float = 5.0  # energy pumping gain [m/(J s^2)]
    a_max: float = 8.0  # max commanded cart acceleration during swing-up [m/s^2]
    kp_center: float = 8.0  # cart centring
    kd_center: float = 1.0
    catch_angle: float = 0.5  # switch to LQR below this |theta| [rad]
    release_angle: float = 0.9  # switch back to swing-up above this |theta| [rad]


class SwingUpLQR(Controller):
    name = "Swing-up + LQR"

    def __init__(self, dt: float, params: CartPoleParams, gains: SwingUpGains | None = None) -> None:
        super().__init__(dt)
        self.p = params
        self.g = gains or SwingUpGains()
        self.lqr = LQR(dt, params)
        self.reset()

    def reset(self) -> None:
        self.mode = "swing-up"
        self.switch_times: list[int] = []
        self._k = 0

    def _force_for_acceleration(self, x: np.ndarray, a: float) -> float:
        """Inverse dynamics: the force that produces cart acceleration `a` in state `x`."""
        p = self.p
        _, vel, th, om = x
        s, c = np.sin(th), np.cos(th)
        ml = p.m * p.l
        th_ddot = (ml * p.g * s - p.d * om - ml * c * a) / p.J
        return (p.M + p.m) * a + ml * c * th_ddot - ml * s * om**2 + p.b * vel

    def compute(self, x: np.ndarray, x_ref: np.ndarray) -> float:
        self._k += 1
        theta = float(wrap_angle(x[2]))

        if self.mode == "swing-up" and abs(theta) < self.g.catch_angle:
            self.mode = "balance"
            self.switch_times.append(self._k)
        elif self.mode == "balance" and abs(theta) > self.g.release_angle:
            self.mode = "swing-up"
            self.switch_times.append(self._k)

        if self.mode == "balance":
            return self.lqr.compute(x, x_ref)

        g = self.g
        E = energy(x, self.p)
        direction = np.sign(x[3] * np.cos(x[2])) or 1.0  # kick-start when hanging perfectly still
        a = g.k_energy * E * direction
        a -= g.kp_center * (x[0] - x_ref[0]) + g.kd_center * x[1]
        a = float(np.clip(a, -g.a_max, g.a_max))
        return self._force_for_acceleration(x, a)
