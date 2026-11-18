"""Cascaded PID: the classical, model-free baseline.

A single PID loop cannot balance the pendulum AND position the cart (one input, two outputs
to regulate). The standard solution is a cascade:

    outer loop (slow, PD):  cart position error  ->  desired pole tilt theta_ref
    inner loop (fast, PID): pole angle error     ->  force on the cart

Physics behind the outer loop: if the inner loop holds the pole at a small tilt theta, the cart
must accelerate with p_ddot ~= g * theta to keep it balanced. Tilting the pole towards the target
therefore drives the cart towards the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pendulum.controllers.base import Controller


@dataclass
class PIDGains:
    kp_theta: float = 60.0
    ki_theta: float = 15.0
    kd_theta: float = 9.0
    kp_pos: float = 0.10  # rad of tilt per metre of position error
    kd_pos: float = 0.16  # rad of tilt per m/s of cart velocity
    max_tilt: float = 0.15  # limit on theta_ref [rad]
    integral_limit: float = 0.5  # anti-windup clamp on the angle integral [rad s]


class CascadedPID(Controller):
    name = "PID (cascaded)"

    def __init__(self, dt: float, gains: PIDGains | None = None, u_max: float = np.inf) -> None:
        super().__init__(dt)
        self.g = gains or PIDGains()
        self.u_max = u_max
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0

    def compute(self, x: np.ndarray, x_ref: np.ndarray) -> float:
        g = self.g
        pos, vel, theta, omega = x

        # Outer loop: lean towards the target, brake with velocity.
        theta_ref = g.kp_pos * (x_ref[0] - pos) - g.kd_pos * (vel - x_ref[1])
        theta_ref = float(np.clip(theta_ref, -g.max_tilt, g.max_tilt))

        # Inner loop. A pole leaning towards +p (theta > 0) is caught by pushing the cart towards +p.
        # The derivative acts on the measured angular velocity, not on the error, so a step in
        # theta_ref does not cause a "derivative kick".
        error = theta - theta_ref
        u_unsat = g.kp_theta * error + g.ki_theta * self._integral + g.kd_theta * omega

        # Conditional integration (anti-windup): stop integrating while the actuator saturates
        # in the same direction as the error.
        saturated = abs(u_unsat) >= self.u_max and np.sign(u_unsat) == np.sign(error)
        if not saturated:
            self._integral = float(np.clip(self._integral + error * self.dt, -g.integral_limit, g.integral_limit))
        return float(u_unsat)
