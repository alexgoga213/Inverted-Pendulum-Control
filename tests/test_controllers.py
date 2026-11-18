import warnings
from dataclasses import replace

import numpy as np
import pytest

from pendulum.controllers import LQR, MPC, CascadedPID, SwingUpLQR
from pendulum.metrics import compute_metrics
from pendulum.model import CartPoleParams, wrap_angle
from pendulum.simulate import push, simulate, step_ref

warnings.filterwarnings("ignore", module="cvxpy")

P = CartPoleParams()
DT = 0.02


def make(name, params=P):
    return {"pid": lambda: CascadedPID(DT, u_max=params.u_max), "lqr": lambda: LQR(DT, params),
            "mpc": lambda: MPC(DT, params, horizon=20)}[name]()


def test_lqr_closed_loop_is_stable():
    assert np.all(np.abs(LQR(DT, P).closed_loop_poles) < 1.0)


@pytest.mark.parametrize("name", ["pid", "lqr", "mpc"])
def test_balances_from_small_tilt(name):
    res = simulate(make(name), P, np.array([0, 0, 0.1, 0]), 6.0)
    assert not res.failed
    assert abs(wrap_angle(res.x[-1, 2])) < 0.01
    assert abs(res.x[-1, 0]) < 0.05


@pytest.mark.parametrize("name", ["pid", "lqr", "mpc"])
def test_tracks_position_step(name):
    res = simulate(make(name), P, np.zeros(4), 8.0, ref=step_ref(0.5, 0.0, 0.5))
    assert not res.failed
    assert res.x[-1, 0] == pytest.approx(0.5, abs=0.03)


@pytest.mark.parametrize("name", ["pid", "lqr", "mpc"])
def test_rejects_push(name):
    res = simulate(make(name), P, np.zeros(4), 6.0, disturbance=push(1.0, 0.2, 10.0))
    assert not res.failed
    assert abs(wrap_angle(res.x[-1, 2])) < 0.02


def test_mpc_respects_actuator_limit_on_its_own():
    weak = replace(P, u_max=5.0)
    mpc = MPC(DT, weak, horizon=20)
    for theta in (0.05, 0.2, 0.3):
        u = mpc.compute(np.array([0, 0, theta, 0]), np.zeros(4))
        assert abs(u) <= weak.u_max + 1e-3


def test_mpc_beats_lqr_with_weak_actuator():
    """The headline result: with |u| <= 5 N and a 17 degree tilt, LQR saturates and loses the pole,
    MPC plans within the limit and recovers."""
    weak = replace(P, u_max=5.0)
    x0 = np.array([0, 0, 0.3, 0])
    assert simulate(LQR(DT, weak), weak, x0, 5.0).failed
    assert not simulate(MPC(DT, weak), weak, x0, 5.0).failed


def test_pid_anti_windup_keeps_integral_bounded():
    pid = CascadedPID(DT, u_max=2.0)
    for _ in range(500):
        pid.compute(np.array([0, 0, 0.3, 0]), np.zeros(4))
    assert abs(pid._integral) <= pid.g.integral_limit + 1e-12


def test_swingup_reaches_upright_and_stays():
    ctrl = SwingUpLQR(DT, P)
    res = simulate(ctrl, P, np.array([0, 0, np.pi, 0]), 10.0, fail_angle=None)
    assert not res.failed
    assert ctrl.mode == "balance"
    assert abs(wrap_angle(res.x[-1, 2])) < 0.01
    assert np.max(np.abs(res.x[:, 0])) < P.track_limit


@pytest.mark.parametrize("p0, dtheta", [(0.1, -0.2), (-0.2, 0.3), (0.2, -0.3), (0.0, 0.15)])
def test_swingup_works_from_different_starts(p0, dtheta):
    ctrl = SwingUpLQR(DT, P)
    res = simulate(ctrl, P, np.array([p0, 0, np.pi + dtheta, 0]), 12.0, fail_angle=None)
    assert not res.failed and ctrl.mode == "balance"
    assert abs(wrap_angle(res.x[-1, 2])) < 0.02


def test_metrics_on_failed_run():
    weak = replace(P, u_max=5.0)
    res = simulate(LQR(DT, weak), weak, np.array([0, 0, 0.3, 0]), 5.0)
    m = compute_metrics(res)
    assert not m.success and m.settling_time != m.settling_time  # nan


def test_simulate_rejects_incompatible_time_steps():
    with pytest.raises(ValueError):
        simulate(LQR(0.015, P), P, np.zeros(4), 1.0, dt_sim=0.004)
