"""Run every experiment and regenerate all figures, animations and metric tables.

    python -m experiments.run_all            # everything (a few minutes, MPC is the slow part)
    python -m experiments.run_all --quick    # skip the operating-envelope sweep and the GIFs
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from pendulum.controllers import LQR, MPC, CascadedPID, SwingUpLQR
from pendulum.metrics import Metrics, compute_metrics, markdown_table
from pendulum.model import CartPoleParams
from pendulum.simulate import constant_ref, push, simulate, step_ref
from pendulum.viz import animate, plot_comparison, plot_region_of_attraction, plot_swingup

warnings.filterwarnings("ignore", module="cvxpy")

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"
DT = 0.02  # controller sample time: 50 Hz
NOMINAL = CartPoleParams()


def controllers_for(design: CartPoleParams):
    """All three controllers, designed for the parameters in `design`."""
    return [CascadedPID(DT, u_max=design.u_max), LQR(DT, design), MPC(DT, design)]


def scenario_step():
    x0 = np.array([0.0, 0.0, 0.2, 0.0])  # pole starts tilted by 11.5 degrees
    ref = step_ref(1.0, 0.0, 1.0)  # then move the cart 1 m to the right
    res = [simulate(c, NOMINAL, x0, 8.0, ref=ref) for c in controllers_for(NOMINAL)]
    mets = [compute_metrics(r, t_from=1.0, step=(1.0, 0.0, 1.0)) for r in res]
    plot_comparison(res, "Tilted start (11.5°) + 1 m position step at t = 1 s", ASSETS / "step_response.png")
    return res, mets


def scenario_push():
    dist = push(1.0, 0.2, 15.0)  # 15 N push for 0.2 s
    res = [simulate(c, NOMINAL, np.zeros(4), 7.0, disturbance=dist) for c in controllers_for(NOMINAL)]
    mets = [compute_metrics(r, t_from=1.0) for r in res]
    plot_comparison(res, "Disturbance rejection: 15 N push for 0.2 s at t = 1 s", ASSETS / "disturbance.png",
                    u_limit=NOMINAL.u_max)
    return res, mets


def scenario_weak_motor():
    weak = replace(NOMINAL, u_max=5.0)  # a 4x weaker actuator
    x0 = np.array([0.0, 0.0, 0.3, 0.0])  # 17 degree initial tilt
    res = [simulate(c, weak, x0, 7.0) for c in controllers_for(weak)]
    mets = [compute_metrics(r) for r in res]
    plot_comparison(res, "Weak actuator (|u| ≤ 5 N), 17° initial tilt", ASSETS / "weak_motor.png", show_ref=False,
                    u_limit=weak.u_max)
    return res, mets


def scenario_robustness():
    # Controllers are designed for the NOMINAL model but the real plant is different, and sensors are noisy.
    plant = NOMINAL.perturbed(m=1.5, l=1.2, b=2.0)  # +50% pole mass, +20% length, 2x friction
    noise = np.array([0.002, 0.01, 0.005, 0.02])  # m, m/s, rad, rad/s (std)
    ref = step_ref(1.0, 0.0, 0.5)
    res = [simulate(c, plant, np.array([0, 0, 0.1, 0]), 8.0, ref=ref, noise_std=noise, seed=1)
           for c in controllers_for(NOMINAL)]
    mets = [compute_metrics(r, t_from=1.0, step=(1.0, 0.0, 0.5)) for r in res]
    plot_comparison(res, "Model mismatch (+50% pole mass, +20% length, 2× friction) + sensor noise",
                    ASSETS / "robustness.png")
    return res, mets


def scenario_swingup():
    ctrl = SwingUpLQR(DT, NOMINAL)
    res = simulate(ctrl, NOMINAL, np.array([0.0, 0.0, np.pi, 0.0]), 10.0, fail_angle=None)
    t_catch = ctrl.switch_times[0] * DT if ctrl.switch_times else float("nan")
    met = compute_metrics(res, angle_from=t_catch if t_catch == t_catch else 0.0)
    plot_swingup(res, ASSETS / "swingup.png")
    return res, met, t_catch


def operating_envelope():
    angles = np.radians(np.arange(5, 46, 5))  # 5..45 deg
    forces = np.array([3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0])
    grid = {}
    for name in ("PID (cascaded)", "LQR", "MPC"):
        ok = np.zeros((len(angles), len(forces)))
        for j, f in enumerate(forces):
            plant = replace(NOMINAL, u_max=float(f))
            ctrl = {c.name: c for c in controllers_for(plant)}[name]
            for i, a in enumerate(angles):
                r = simulate(ctrl, plant, np.array([0, 0, a, 0]), 6.0)
                ok[i, j] = 0.0 if r.failed else 1.0
        grid[name] = ok
        print(f"  envelope {name:<15} {ok.mean() * 100:5.1f}% recovered")
    plot_region_of_attraction(grid, np.degrees(angles), forces, ASSETS / "operating_envelope.png")
    return {k: float(v.mean()) for k, v in grid.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ASSETS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    t0 = time.perf_counter()

    lqr = LQR(DT, NOMINAL)
    print(f"LQR gain K = {np.round(lqr.K, 2).tolist()}")
    print(f"closed-loop poles |z| = {np.round(np.abs(lqr.closed_loop_poles), 4).tolist()}")

    tables: dict[str, list[Metrics]] = {}
    print("[1/6] step response");      step_res, tables["step"] = scenario_step()
    print("[2/6] push disturbance");   push_res, tables["push"] = scenario_push()
    print("[3/6] weak actuator");      weak_res, tables["weak_motor"] = scenario_weak_motor()
    print("[4/6] robustness");         _, tables["robustness"] = scenario_robustness()
    print("[5/6] swing-up");           swing_res, swing_met, t_catch = scenario_swingup()
    tables["swingup"] = [swing_met]

    envelope = None
    if not args.quick:
        print("[6/6] operating envelope (slow)")
        envelope = operating_envelope()
        print("rendering GIFs...")
        animate(weak_res, ASSETS / "weak_motor.gif", "Weak actuator (|u| ≤ 5 N), 17° initial tilt", max_seconds=5.0)
        animate(push_res, ASSETS / "disturbance.gif", "15 N push at t = 1 s", max_seconds=5.0)
        animate([swing_res], ASSETS / "swingup.gif", "Energy-based swing-up + LQR catch", max_seconds=7.0)

    md = []
    titles = {
        "step": "Tilted start + 1 m step", "push": "15 N push", "weak_motor": "Weak actuator (5 N), 17° tilt",
        "robustness": "Model mismatch + noise", "swingup": f"Swing-up (caught after {t_catch:.2f} s)",
    }
    for key, rows in tables.items():
        md.append(f"### {titles[key]}\n\n{markdown_table(rows)}\n")
    if envelope:
        md.append("### Operating envelope (share of initial angles × actuator limits recovered)\n\n"
                  "| Controller | Recovered |\n|---|---|\n"
                  + "\n".join(f"| {k} | {v * 100:.0f}% |" for k, v in envelope.items()) + "\n")
    (RESULTS / "metrics.md").write_text("\n".join(md), encoding="utf-8")
    (RESULTS / "metrics.json").write_text(
        json.dumps({k: [m.as_dict() for m in v] for k, v in tables.items()} | {"envelope": envelope,
                   "swingup_catch_time": t_catch, "lqr_gain": lqr.K.tolist()}, indent=2, default=float),
        encoding="utf-8",
    )
    print("\n" + "\n".join(md))
    print(f"done in {time.perf_counter() - t0:.0f}s -> {ASSETS}, {RESULTS}")


if __name__ == "__main__":
    main()
