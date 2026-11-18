"""Closed-loop simulation: digital controller (ZOH) + continuous nonlinear plant (RK4).

Realistic effects that can be switched on per scenario:
* actuator saturation |u| <= u_max
* Gaussian measurement noise on every state
* external force disturbances (e.g. a push on the cart)
* model mismatch: the plant may use different parameters than the controller was designed with
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pendulum.controllers.base import Controller
from pendulum.model import CartPoleParams, rk4_step, wrap_angle

RefFn = Callable[[float], np.ndarray]
DisturbanceFn = Callable[[float], float]


def constant_ref(position: float = 0.0) -> RefFn:
    r = np.array([position, 0.0, 0.0, 0.0])
    return lambda t: r


def step_ref(t_step: float, before: float, after: float) -> RefFn:
    a, b = np.array([before, 0, 0, 0.0]), np.array([after, 0, 0, 0.0])
    return lambda t: b if t >= t_step else a


def push(t_start: float, duration: float, force: float) -> DisturbanceFn:
    """A rectangular force pulse on the cart (e.g. someone pushes it)."""
    return lambda t: force if t_start <= t < t_start + duration else 0.0


@dataclass
class SimResult:
    name: str
    t: np.ndarray
    x: np.ndarray  # (N, 4) true states
    u: np.ndarray  # (N,) applied (saturated) force
    ref: np.ndarray  # (N, 4)
    disturbance: np.ndarray  # (N,)
    params: CartPoleParams
    compute_times: np.ndarray = field(default_factory=lambda: np.array([]))
    failed: bool = False
    failure_reason: str = ""
    modes: list[str] | None = None


def simulate(
    controller: Controller,
    plant: CartPoleParams,
    x0: np.ndarray,
    duration: float,
    ref: RefFn | None = None,
    disturbance: DisturbanceFn | None = None,
    noise_std: np.ndarray | float = 0.0,
    dt_sim: float = 0.001,
    seed: int = 0,
    fail_angle: float | None = 1.2,
) -> SimResult:
    """Run one closed-loop simulation.

    `fail_angle`: stop early if |theta| exceeds it (pole has fallen). None disables the check
    (needed for swing-up, which starts hanging down).
    """
    ref = ref or constant_ref()
    disturbance = disturbance or (lambda t: 0.0)
    rng = np.random.default_rng(seed)
    noise_std = np.broadcast_to(np.asarray(noise_std, dtype=float), (4,))

    substeps = int(round(controller.dt / dt_sim))
    if substeps < 1 or abs(substeps * dt_sim - controller.dt) > 1e-12:
        raise ValueError("controller.dt must be an integer multiple of dt_sim")
    n_ctrl = int(round(duration / controller.dt))

    controller.reset()
    x = np.asarray(x0, dtype=float).copy()
    ts, xs, us, refs, dists, ctimes, modes = [], [], [], [], [], [], []
    failed, reason = False, ""

    for k in range(n_ctrl):
        t = k * controller.dt
        r = ref(t)
        y = x + rng.normal(0.0, noise_std)  # measurement

        t0 = time.perf_counter()
        u = controller.compute(y, r)
        ctimes.append(time.perf_counter() - t0)
        u = float(np.clip(u, -plant.u_max, plant.u_max))
        w = disturbance(t)

        ts.append(t), xs.append(x.copy()), us.append(u), refs.append(r), dists.append(w)
        if hasattr(controller, "mode"):
            modes.append(controller.mode)

        for _ in range(substeps):
            x = rk4_step(x, u + w, dt_sim, plant)

        if fail_angle is not None and abs(wrap_angle(x[2])) > fail_angle:
            failed, reason = True, "pole fell"
        elif abs(x[0]) > plant.track_limit:
            failed, reason = True, "cart left the track"
        if failed:
            break

    return SimResult(
        name=controller.name,
        t=np.array(ts),
        x=np.array(xs),
        u=np.array(us),
        ref=np.array(refs),
        disturbance=np.array(dists),
        params=plant,
        compute_times=np.array(ctimes),
        failed=failed,
        failure_reason=reason,
        modes=modes or None,
    )
