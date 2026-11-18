### Tilted start + 1 m step

| Controller | Result | Settling time [s] | Max angle [°] | Overshoot [%] | Max force [N] | Saturated [%] | Control energy [N²s] | Compute [ms] |
|---|---|---|---|---|---|---|---|---|
| PID (cascaded) | ✅ | 4.34 | 11.5 | 0.0 | 12.0 | 0.0 | 14.8 | 0.011 |
| LQR | ✅ | 2.32 | 12.5 | 1.9 | 16.2 | 0.0 | 27.0 | 0.005 |
| MPC | ✅ | 2.32 | 12.5 | 2.0 | 16.2 | 0.0 | 27.0 | 12.660 |

### 15 N push

| Controller | Result | Settling time [s] | Max angle [°] | Overshoot [%] | Max force [N] | Saturated [%] | Control energy [N²s] | Compute [ms] |
|---|---|---|---|---|---|---|---|---|
| PID (cascaded) | ✅ | 3.18 | 18.3 | 0.0 | 20.0 | 1.7 | 88.2 | 0.010 |
| LQR | ✅ | 1.84 | 11.2 | 0.0 | 20.0 | 1.4 | 72.9 | 0.008 |
| MPC | ✅ | 1.84 | 11.2 | 0.0 | 20.0 | 1.4 | 72.9 | 10.429 |

### Weak actuator (5 N), 17° tilt

| Controller | Result | Settling time [s] | Max angle [°] | Overshoot [%] | Max force [N] | Saturated [%] | Control energy [N²s] | Compute [ms] |
|---|---|---|---|---|---|---|---|---|
| PID (cascaded) | ✅ | 4.40 | 17.2 | 0.0 | 5.0 | 4.6 | 13.3 | 0.010 |
| LQR | ❌ | — | 65.5 | 0.0 | 5.0 | 93.1 | 28.0 | 0.005 |
| MPC | ✅ | 2.56 | 17.2 | 0.0 | 5.0 | 6.3 | 21.0 | 9.088 |

### Model mismatch + noise

| Controller | Result | Settling time [s] | Max angle [°] | Overshoot [%] | Max force [N] | Saturated [%] | Control energy [N²s] | Compute [ms] |
|---|---|---|---|---|---|---|---|---|
| PID (cascaded) | ✅ | 3.16 | 5.7 | 0.0 | 5.9 | 0.0 | 7.8 | 0.011 |
| LQR | ✅ | 1.94 | 7.2 | 3.4 | 8.0 | 0.0 | 13.1 | 0.006 |
| MPC | ✅ | 1.94 | 7.2 | 3.3 | 8.0 | 0.0 | 13.0 | 34.307 |

### Swing-up (caught after 2.02 s)

| Controller | Result | Settling time [s] | Max angle [°] | Overshoot [%] | Max force [N] | Saturated [%] | Control energy [N²s] | Compute [ms] |
|---|---|---|---|---|---|---|---|---|
| Swing-up + LQR | ✅ | 3.24 | 27.3 | 0.0 | 18.8 | 0.0 | 71.4 | 0.009 |

### Operating envelope (share of initial angles × actuator limits recovered)

| Controller | Recovered |
|---|---|
| PID (cascaded) | 54% |
| LQR | 47% |
| MPC | 53% |
