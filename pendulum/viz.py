"""Plots and GIF animations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from pendulum.model import wrap_angle  # noqa: E402
from pendulum.simulate import SimResult  # noqa: E402

COLORS = {"PID (cascaded)": "#E07A1F", "LQR": "#2E6FD8", "MPC": "#2BA84A", "Swing-up + LQR": "#8E44AD"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def _color(name: str) -> str:
    return COLORS.get(name, "#555555")


def plot_comparison(results: list[SimResult], title: str, path: Path, show_ref: bool = True,
                    u_limit: float | None = None) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for r in results:
        c = _color(r.name)
        label = r.name + ("  ✗ " + r.failure_reason if r.failed else "")
        axes[0].plot(r.t, r.x[:, 0], color=c, lw=2, label=label)
        axes[1].plot(r.t, np.degrees(wrap_angle(r.x[:, 2])), color=c, lw=2)
        axes[2].plot(r.t, r.u, color=c, lw=1.6)
        if r.failed:
            for ax, y in zip(axes, (r.x[-1, 0], np.degrees(wrap_angle(r.x[-1, 2])), r.u[-1])):
                ax.plot(r.t[-1], y, "x", color=c, ms=10, mew=3)
    if show_ref:
        longest = max(results, key=lambda r: len(r.t))
        axes[0].plot(longest.t, longest.ref[:, 0], "k--", lw=1, label="reference")
    if u_limit is not None:
        for s in (-1, 1):
            axes[2].axhline(s * u_limit, color="r", ls=":", lw=1.2)
        axes[2].text(0.0, u_limit, " actuator limit", color="r", va="bottom", fontsize=8)
    axes[0].set_ylabel("cart position [m]")
    axes[1].set_ylabel("pole angle [°]")
    axes[2].set_ylabel("force [N]")
    axes[2].set_xlabel("time [s]")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_swingup(r: SimResult, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    theta = np.unwrap(r.x[:, 2])
    deg = np.degrees(theta)
    axes[0].plot(r.t, deg, color=_color(r.name), lw=2)
    lo, hi = np.floor(deg.min() / 180) * 180, np.ceil(deg.max() / 180) * 180
    for level in np.arange(lo, hi + 1, 180):
        upright = int(round(level / 180)) % 2 == 0
        axes[0].axhline(level, color="k", lw=0.8, ls="--" if upright else ":")
        axes[0].text(r.t[-1], level, " upright" if upright else " hanging", va="center", fontsize=8)
    axes[0].set_yticks(np.arange(lo, hi + 1, 90))
    axes[0].set_ylabel("pole angle [°]")
    axes[1].plot(r.t, r.x[:, 0], color="#333", lw=2)
    axes[1].set_ylabel("cart position [m]")
    axes[2].plot(r.t, r.u, color="#333", lw=1.4)
    axes[2].set_ylabel("force [N]")
    axes[2].set_xlabel("time [s]")
    if r.modes:
        balance = np.array([m == "balance" for m in r.modes])
        if balance.any():
            t_switch = r.t[np.argmax(balance)]
            for ax in axes:
                ax.axvspan(t_switch, r.t[-1], color="#2E6FD8", alpha=0.08)
                ax.axvline(t_switch, color="#2E6FD8", lw=1.2)
            tr = axes[0].get_xaxis_transform()  # x in data units, y in axes fraction
            axes[0].text(t_switch, 0.7, "  LQR takes over", color="#2E6FD8", transform=tr,
                         bbox=dict(fc="white", ec="none", alpha=0.8))
            axes[0].text(0.05, 0.7, "energy pumping", color="#8E44AD", transform=tr,
                         bbox=dict(fc="white", ec="none", alpha=0.8))
    fig.suptitle("Swing-up: from hanging down to balanced", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_region_of_attraction(grid: dict[str, np.ndarray], angles_deg: np.ndarray, forces: np.ndarray,
                              path: Path) -> None:
    names = list(grid)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        ok = grid[name]
        ax.imshow(ok.T, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                  extent=[angles_deg[0], angles_deg[-1], forces[0], forces[-1]], interpolation="nearest")
        ax.set_title(f"{name}\n{ok.mean() * 100:.0f}% of cases recovered", fontsize=10)
        ax.set_xlabel("initial pole angle [°]")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("actuator limit u_max [N]")
    fig.suptitle("Operating envelope: green = balanced, red = fell / left the track", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def animate(results: list[SimResult], path: Path, title: str, fps: int = 25, max_seconds: float | None = None,
            track_limit: float | None = None) -> None:
    """Stacked animation, one cart-pole per controller."""
    n = len(results)
    dt = results[0].t[1] - results[0].t[0]
    stride = max(1, int(round(1.0 / (fps * dt))))
    t_end = max(r.t[-1] for r in results)
    if max_seconds:
        t_end = min(t_end, max_seconds)
    frames = int(t_end / (dt * stride)) + 1

    L = 2 * results[0].params.l  # full pole length for drawing
    lim = track_limit or results[0].params.track_limit
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.1 * n + 0.4), squeeze=False)
    artists = []
    for ax, r in zip(axes[:, 0], results):
        ax.set_xlim(-lim - 0.3, lim + 0.3)
        ax.set_ylim(-0.75, 0.75)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.plot([-lim, lim], [-0.08, -0.08], color="#444", lw=3)
        for s in (-1, 1):
            ax.plot([s * lim] * 2, [-0.14, 0.02], color="#c0392b", lw=3)
        ref_mark, = ax.plot([], [], "v", color="k", ms=7)
        cart = Rectangle((0, 0), 0.3, 0.14, fc=_color(r.name), ec="k", lw=1)
        ax.add_patch(cart)
        pole, = ax.plot([], [], color="#222", lw=4, solid_capstyle="round")
        bob, = ax.plot([], [], "o", color="#222", ms=5)
        label = ax.text(-lim - 0.25, 0.6, r.name, fontsize=10, fontweight="bold", color=_color(r.name))
        status = ax.text(lim + 0.25, 0.6, "", fontsize=9, ha="right")
        push_arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2.5))
        artists.append((r, cart, pole, bob, status, ref_mark, push_arrow, label))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    time_txt = fig.text(0.5, 0.01, "", ha="center", fontsize=9)

    def update(i: int):
        k_global = i * stride
        time_txt.set_text(f"t = {k_global * dt:4.2f} s")
        out = [time_txt]
        for r, cart, pole, bob, status, ref_mark, push_arrow, _ in artists:
            k = min(k_global, len(r.t) - 1)
            p, _, th, _ = r.x[k]
            cart.set_xy((p - 0.15, -0.07))
            tip = (p + L * np.sin(th), 0.07 + L * np.cos(th))
            pole.set_data([p, tip[0]], [0.07, tip[1]])
            bob.set_data([tip[0]], [tip[1]])
            ref_mark.set_data([r.ref[k, 0]], [0.32])
            w = r.disturbance[k]
            if abs(w) > 0:
                push_arrow.xy, push_arrow.xyann = (p - 0.15 * np.sign(w), 0.0), (p - 0.15 * np.sign(w) - 0.35 * np.sign(w), 0.0)
                push_arrow.set_visible(True)
            else:
                push_arrow.set_visible(False)
            if r.failed and k_global >= len(r.t) - 1:
                status.set_text("✗ " + r.failure_reason.upper())
                status.set_color("#c0392b")
            elif r.modes:
                status.set_text(r.modes[k])
                status.set_color("#555")
            else:
                status.set_text(f"{np.degrees(wrap_angle(th)):+5.1f}°  |  {r.u[k]:+5.1f} N")
                status.set_color("#555")
            out += [cart, pole, bob, status, ref_mark, push_arrow]
        return out

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)
