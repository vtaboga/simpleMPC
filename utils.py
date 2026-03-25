"""Figure generation for the building MPC simulation."""

import numpy as np
import matplotlib.pyplot as plt


def _format_day_axis(ax, n_days, day_names):
    ax.set_xlim(0, n_days * 24)
    ax.set_xticks(np.arange(n_days + 1) * 24)
    ax.set_xticklabels(day_names[: n_days + 1])


def _shade_dr(ax, dr_days):
    """Highlight the DR event window."""
    for dr_day in dr_days:
        ax.axvspan(dr_day * 24 + 16, dr_day * 24 + 20,
                   color="red", alpha=0.08, label="DR event")


def _shade_occupied(ax, T_min, t):
    """Light blue background during occupied hours."""
    occ = T_min > 18
    for i in range(len(t) - 1):
        if occ[i]:
            ax.axvspan(t[i], t[i + 1], color="#2196F3", alpha=0.04)


def plot_mpc_overview(t_sim, scenario, x_mpc, Q_mpc, *, n_steps, n_days, day_names, dr_days,
                      q_sol_act=None):
    """Figure 1 – scenario context and MPC results.

    If ``q_sol_act`` is given, solar **actual** (plant) is filled; nominal
    forecast is drawn as a dashed line (MPC uses nominal in its horizon).
    """
    _, T_ext, Q_sol, Q_int, prices, T_min, T_max, T_ref = scenario
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("MPC Simulation Overview – Winter Heating Week", fontsize=14, y=0.98)

    # (a) Weather
    ax = axes[0]
    ax.plot(t_sim, T_ext[:n_steps], color="#757575", linewidth=0.9)
    ax.set_ylabel("Outdoor T  [°C]")
    ax.set_title("(a) Outdoor temperature & solar (nominal vs actual)", loc="left", fontsize=10)
    ax2 = ax.twinx()
    if q_sol_act is not None:
        ax2.fill_between(t_sim, 0, q_sol_act[:n_steps], alpha=0.35, color="#FFC107", step="post")
        ax2.step(t_sim, Q_sol[:n_steps], where="post", color="#E65100", linewidth=0.8,
                 linestyle="--", alpha=0.85, label="Q_sol forecast")
        ax2.legend(loc="upper right", fontsize=7)
    else:
        ax2.fill_between(t_sim, 0, Q_sol[:n_steps], alpha=0.35, color="#FFC107")
    ax2.set_ylabel("Solar gain  [kW]", color="#FFC107")
    ax2.set_ylim(0, 4)
    _shade_dr(ax, dr_days)
    _format_day_axis(ax, n_days, day_names)

    # (b) Electricity price
    ax = axes[1]
    ax.step(t_sim, prices[:n_steps], where="post", color="#4CAF50", linewidth=1)
    ax.set_ylabel("Price  [$/kWh]")
    ax.set_title("(b) Electricity price (DR event shaded)", loc="left", fontsize=10)
    _shade_dr(ax, dr_days)
    _format_day_axis(ax, n_days, day_names)

    # (c) Zone temperatures
    ax = axes[2]
    _shade_occupied(ax, T_min, t_sim)
    ax.plot(t_sim, x_mpc[:n_steps, 0], color="#2196F3", linewidth=1.1, label="T_indoor")
    ax.plot(t_sim, x_mpc[:n_steps, 1], color="#FF9800", linewidth=0.9, label="T_wall")
    ax.step(t_sim, T_ref[:n_steps], where="post", color="#4CAF50",
            linewidth=1.0, linestyle="-.", alpha=0.7, label="T_ref")
    ax.step(t_sim, T_min[:n_steps], where="post", color="grey",
            linewidth=0.7, linestyle="--", alpha=0.5, label="T_min / T_max")
    ax.step(t_sim, T_max[:n_steps], where="post", color="grey",
            linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_ylabel("Temperature  [°C]")
    ax.set_ylim(12, 27)
    ax.set_title("(c) Zone temperatures – MPC", loc="left", fontsize=10)
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    _shade_dr(ax, dr_days)
    _format_day_axis(ax, n_days, day_names)

    # (d) Heating power
    ax = axes[3]
    ax.fill_between(t_sim, 0, Q_mpc, alpha=0.5, color="#E53935", step="post")
    ax.step(t_sim, Q_mpc, where="post", color="#E53935", linewidth=0.7)
    ax.set_ylabel("Heating  [kW]")
    ax.set_xlabel("Day of week")
    ax.set_title("(d) Heating power – MPC", loc="left", fontsize=10)
    _shade_dr(ax, dr_days)
    _format_day_axis(ax, n_days, day_names)

    fig.tight_layout()
    return fig


def plot_comparison(t_sim, scenario, x_mpc, Q_mpc, x_base, Q_base, m_mpc, m_base, *,
                    n_steps, n_days, day_names, dr_days, dt):
    """Figure 2 – MPC vs baseline side-by-side."""
    _, T_ext, Q_sol, Q_int, prices, T_min, T_max, T_ref = scenario
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("MPC vs Baseline (Thermostat) – Comparison", fontsize=14, y=0.98)

    # (a) Indoor temperature
    ax = axes[0]
    _shade_occupied(ax, T_min, t_sim)
    ax.plot(t_sim, x_mpc[:n_steps, 0], color="#2196F3", linewidth=1, label="MPC")
    ax.plot(t_sim, x_base[:n_steps, 0], color="#FF5722", linewidth=1, alpha=0.8, label="Baseline")
    ax.step(t_sim, T_ref[:n_steps], where="post", color="#4CAF50",
            linewidth=0.9, linestyle="-.", alpha=0.6, label="T_ref")
    ax.step(t_sim, T_min[:n_steps], where="post", color="k",
            linewidth=0.7, linestyle="--", alpha=0.3, label="T_min / T_max")
    ax.step(t_sim, T_max[:n_steps], where="post", color="k",
            linewidth=0.7, linestyle="--", alpha=0.3)
    ax.set_ylabel("Indoor T  [°C]")
    ax.set_ylim(12, 27)
    ax.set_title("(a) Indoor air temperature", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    _shade_dr(ax, dr_days)
    _format_day_axis(ax, n_days, day_names)

    # (b) Heating power + price overlay
    ax = axes[1]
    ax.step(t_sim, Q_mpc, where="post", color="#2196F3", linewidth=0.9, label="MPC")
    ax.step(t_sim, Q_base, where="post", color="#FF5722", linewidth=0.9, alpha=0.7, label="Baseline")
    ax.set_ylabel("Heating  [kW]")
    ax.set_title("(b) Heating power", loc="left", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax3 = ax.twinx()
    ax3.step(t_sim, prices[:n_steps], where="post", color="#4CAF50",
             linewidth=0.6, alpha=0.4)
    ax3.set_ylabel("Price [$/kWh]", color="#4CAF50", fontsize=8)
    _shade_dr(ax, dr_days)
    _format_day_axis(ax, n_days, day_names)

    # (c) Cumulative cost
    ax = axes[2]
    cost_mpc  = np.cumsum(Q_mpc * prices[:n_steps]) * dt
    cost_base = np.cumsum(Q_base * prices[:n_steps]) * dt
    ax.plot(t_sim, cost_mpc, color="#2196F3", linewidth=1.2,
            label=f"MPC  (${m_mpc['cost']:.2f})")
    ax.plot(t_sim, cost_base, color="#FF5722", linewidth=1.2,
            label=f"Baseline  (${m_base['cost']:.2f})")
    ax.set_ylabel("Cumulative cost  [$]")
    ax.set_xlabel("Day of week")
    ax.set_title("(c) Cumulative electricity cost", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    _shade_dr(ax, dr_days)
    _format_day_axis(ax, n_days, day_names)

    fig.tight_layout()
    return fig


def plot_dr_day(t_sim, scenario, x_mpc, Q_mpc, x_base, Q_base, *, n_steps, dr_detail_day):
    """Figure 3 – zoomed view of one DR day."""
    _, T_ext, Q_sol, Q_int, prices, T_min, T_max, T_ref = scenario
    t0h = dr_detail_day * 24
    t1h = (dr_detail_day + 1) * 24
    sl = (t_sim >= t0h) & (t_sim < t1h)
    t_day = t_sim[sl]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Demand-Response Day Detail (Thursday)", fontsize=13, y=0.98)

    # (a) Price + outdoor temp
    ax = axes[0]
    ax.step(t_day, prices[:n_steps][sl], where="post", color="#4CAF50", linewidth=1.2)
    ax.set_ylabel("Price  [$/kWh]")
    ax.axvspan(t0h + 16, t0h + 20, color="red", alpha=0.10, label="DR event")
    ax.legend(fontsize=8, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(t_day, T_ext[:n_steps][sl], color="#757575", linewidth=0.8, linestyle=":")
    ax2.set_ylabel("T_ext [°C]", color="#757575", fontsize=8)

    # (b) Temperature
    ax = axes[1]
    ax.plot(t_day, x_mpc[:n_steps, 0][sl], color="#2196F3", linewidth=1.2, label="MPC  T_i")
    ax.plot(t_day, x_mpc[:n_steps, 1][sl], color="#2196F3", linewidth=0.7,
            linestyle=":", alpha=0.6, label="MPC  T_wall")
    ax.plot(t_day, x_base[:n_steps, 0][sl], color="#FF5722", linewidth=1.2, label="Base T_i")
    ax.step(t_day, T_ref[:n_steps][sl], where="post", color="#4CAF50",
            linestyle="-.", linewidth=0.9, alpha=0.6, label="T_ref")
    ax.step(t_day, T_min[:n_steps][sl], where="post", color="k",
            linestyle="--", linewidth=0.7, alpha=0.4, label="T_min")
    ax.set_ylabel("Temperature  [°C]")
    ax.set_ylim(12, 27)
    ax.axvspan(t0h + 16, t0h + 20, color="red", alpha=0.08)
    ax.legend(fontsize=7, ncol=4, loc="upper right")

    # (c) Heating power
    ax = axes[2]
    ax.step(t_day, Q_mpc[sl], where="post", color="#2196F3", linewidth=1, label="MPC")
    ax.step(t_day, Q_base[sl], where="post", color="#FF5722", linewidth=1, label="Baseline")
    ax.set_ylabel("Heating  [kW]")
    ax.set_xlabel("Hour of day")
    ax.axvspan(t0h + 16, t0h + 20, color="red", alpha=0.08)
    ax.legend(fontsize=8)

    for a in axes:
        hticks = np.arange(t0h, t1h + 1, 2)
        a.set_xticks(hticks)
        a.set_xticklabels([f"{int(h % 24):02d}:00" for h in hticks], fontsize=8)
        a.set_xlim(t0h, t1h)

    fig.tight_layout()
    return fig


def plot_mpc_replan_thursday(scenario, thursday_replans, Q_mpc, *, dt, dr_detail_day):
    """MPC **receding-horizon** planned heating on Thursday: one curve per solve.

    Each coloured **step** curve is the full open-loop plan
    ``(Q_k, Q_{k+1}, …, Q_{k+N-1})`` from the optimisation at step ``k``.
    Re-solving shifts these plans as the state and short-term forecast evolve.
    The **black** curve is the heating actually applied that day (first move
    of each plan).
    """
    _, _, _, _, prices, _, _, _ = scenario
    t0_abs = dr_detail_day * 24.0
    t1_abs = (dr_detail_day + 1) * 24.0
    k0 = int(round(t0_abs / dt))
    k1 = int(round(t1_abs / dt))

    fig, (ax_u, ax_p) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    fig.suptitle(
        "MPC action replanning — Thursday (each colour = one optimisation solve)",
        fontsize=13,
        y=0.98,
    )

    n = len(thursday_replans)
    if n == 0:
        ax_u.text(0.5, 0.5, "No replan data for this day.", ha="center", va="center", transform=ax_u.transAxes)
        fig.tight_layout()
        return fig

    ks = np.array([tp[0] for tp in thursday_replans], dtype=float)
    norm = plt.Normalize(vmin=ks.min(), vmax=ks.max())
    cmap = plt.cm.plasma

    for k, Qp in thursday_replans:
        nh = len(Qp)
        t_seg = (k + np.arange(nh)) * dt
        color = cmap(norm(k))
        ax_u.step(
            t_seg, Qp, where="post", color=color, linewidth=1.0, alpha=0.45,
        )

    t_app = np.arange(k0, k1) * dt
    ax_u.axvspan(t0_abs + 16, t0_abs + 20, color="red", alpha=0.12, zorder=0)
    ax_u.set_ylabel("Planned / applied heating  [kW]")
    ax_u.set_ylim(bottom=0)
    ax_u.set_title(
        "(a) Open-loop plans (faded) vs realised first move each step (black)",
        loc="left",
        fontsize=10,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_u, fraction=0.035, pad=0.02)
    cbar.set_label("MPC solve step index $k$ (Thu)")

    # Price context for the same day
    ax_p.step(
        t_app, prices[k0:k1], where="post", color="#2E7D32", linewidth=1.1,
    )
    ax_p.axvspan(t0_abs + 16, t0_abs + 20, color="red", alpha=0.10)
    ax_p.set_ylabel("Price  [$/kWh]")
    ax_p.set_xlabel("Hour of week from Mon 00:00  [h]  (Thu = 72–96)")
    ax_p.set_title("(b) Electricity price (DR window shaded)", loc="left", fontsize=10)
    ax_p.set_xlim(t0_abs, t1_abs)

    # Secondary x-axis: local hour of day on Thursday
    ax2 = ax_p.secondary_xaxis(
        "top",
        functions=(lambda x, t0=t0_abs: x - t0, lambda x, t0=t0_abs: x + t0),
    )
    ax2.set_xlabel("Thursday — hour of day  [h]")

    fig.tight_layout()
    return fig
