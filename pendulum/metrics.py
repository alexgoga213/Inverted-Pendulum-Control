"""Performance metrics computed from a simulation result."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from pendulum.model import wrap_angle
from pendulum.simulate import SimResult


@dataclass
class Metrics:
    controller: str
    success: bool
    settling_time: float  # [s] after which the system stays within tolerance (nan if never)
    max_angle_deg: float
    overshoot_pct: float  # cart position overshoot for a step reference
    max_cart_travel: float  # [m]
    control_energy: float  # integral of u^2 dt [N^2 s]
    max_force: float  # [N]
    saturation_pct: float  # % of samples at the actuator limit
    mean_compute_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def settling_time(res: SimResult, t_from: float = 0.0, pos_tol: float = 0.05, ang_tol: float = 0.02) -> float:
    """Time (measured from t_from) after which |p - p_ref| < pos_tol and |theta| < ang_tol forever."""
    if res.failed:
        return float("nan")
    ok = (np.abs(res.x[:, 0] - res.ref[:, 0]) < pos_tol) & (np.abs(wrap_angle(res.x[:, 2])) < ang_tol)
    ok &= res.t >= t_from
    if not ok[-1]:
        return float("nan")
    bad = np.where(~ok)[0]
    last_bad = bad[-1] if len(bad) else -1
    if last_bad + 1 >= len(res.t):
        return float("nan")
    return float(res.t[last_bad + 1] - t_from)


def overshoot(res: SimResult, t_step: float, start: float, target: float) -> float:
    """Cart position overshoot (in % of the step size) after a step at t_step."""
    span = target - start
    if span == 0:
        return 0.0
    after = res.x[res.t >= t_step, 0]
    if after.size == 0:
        return float("nan")
    peak = np.max((after - start) / span)
    return float(max(0.0, peak - 1.0) * 100.0)


def compute_metrics(
    res: SimResult,
    t_from: float = 0.0,
    step: tuple[float, float, float] | None = None,
    angle_from: float = 0.0,
) -> Metrics:
    """`step` = (t_step, start, target) enables the overshoot metric.
    `angle_from` ignores the max-angle metric before that time (e.g. during swing-up)."""
    dt = res.t[1] - res.t[0] if len(res.t) > 1 else 0.0
    u_max = res.params.u_max
    mask = res.t >= angle_from
    max_angle = np.degrees(np.max(np.abs(wrap_angle(res.x[mask, 2])))) if mask.any() else float("nan")
    return Metrics(
        controller=res.name,
        success=not res.failed,
        settling_time=settling_time(res, t_from),
        max_angle_deg=float(max_angle),
        overshoot_pct=overshoot(res, *step) if step else 0.0,
        max_cart_travel=float(np.max(np.abs(res.x[:, 0]))),
        control_energy=float(np.sum(res.u**2) * dt),
        max_force=float(np.max(np.abs(res.u))),
        saturation_pct=float(np.mean(np.abs(res.u) >= u_max - 1e-9) * 100.0),
        mean_compute_ms=float(np.mean(res.compute_times) * 1e3) if res.compute_times.size else 0.0,
    )


def markdown_table(rows: list[Metrics]) -> str:
    def fmt(v: float, nd: int = 2) -> str:
        return "—" if v != v else f"{v:.{nd}f}"  # nan -> em dash

    lines = [
        "| Controller | Result | Settling time [s] | Max angle [°] | Overshoot [%] | Max force [N] "
        "| Saturated [%] | Control energy [N²s] | Compute [ms] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in rows:
        lines.append(
            f"| {m.controller} | {'✅' if m.success else '❌'} | {fmt(m.settling_time)} | {fmt(m.max_angle_deg, 1)} "
            f"| {fmt(m.overshoot_pct, 1)} | {fmt(m.max_force, 1)} | {fmt(m.saturation_pct, 1)} "
            f"| {fmt(m.control_energy, 1)} | {fmt(m.mean_compute_ms, 3)} |"
        )
    return "\n".join(lines)
