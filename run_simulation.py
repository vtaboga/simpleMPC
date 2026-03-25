"""
MPC vs Rule-Based Control for Building Heating – Demand Response Scenario
=========================================================================

Simulates a **one-week winter heating scenario** comparing two strategies:

1. **MPC** – optimal control that exploits the building's thermal
   mass to shift heating away from expensive hours.
2. **Baseline** – a simple proportional thermostat that tracks a comfort
   setpoint regardless of electricity prices.

The week includes time-of-use electricity pricing and a **demand-response
(DR) event** on Thursday afternoon where the price spikes to $0.60/kWh,
testing the MPC's ability to pre-heat using cheap off-peak electricity.

**Forecast mismatch:** Solar and internal gains used inside the MPC horizon
are the **nominal** profiles from ``generate_scenario``; the simulated
building evolves with **independently perturbed** gains (Gaussian noise,
reproducible seed) so the controller does not see perfect disturbance
forecasts.

Usage
-----
    python run_simulation.py

Outputs
    • Console summary of energy, cost, and comfort metrics
    • fig_mpc_overview.png        – scenario inputs & MPC behaviour
    • fig_comparison.png          – side-by-side MPC vs baseline
    • fig_dr_day_detail.png       – zoom on one DR day
    • fig_mpc_replan_thursday.png – MPC planned heating at each solve (Thursday)

"""

import time
import numpy as np
import matplotlib.pyplot as plt

import utils
from building_model import DEFAULT_PARAMS, discretize, simulate_step, print_model_info
from mpc_controller import BuildingMPC

# ── Simulation settings ────────────────────────────────────────────────────
DT = 0.25          # Time step  [hours]  (15 minutes)
N_DAYS = 7         # Simulation length   [days]
N_HORIZON = 24     # MPC look-ahead: 6 hours  [steps]
N_STEPS = int(N_DAYS * 24 / DT)   # 672 steps total
DR_DAYS = [1, 3, 4]         # Demand-response event days (0 = Mon)
DR_DETAIL_DAY = 3           # Day index for zoomed DR figure (3 = Thu)

# Plant vs forecast: MPC uses nominal solar/internal from generate_scenario;
# the simulated building sees independent Gaussian perturbations on these gains.
RNG_SEED = 42
GAIN_NOISE_SOL_REL = 0.12   # std of solar noise ≈ this fraction × (|Q_sol| + floor)
GAIN_NOISE_SOL_ABS = 0.06   # kW, floor std when Q_sol ≈ 0 (cloud / sensor noise)
GAIN_NOISE_INT_REL = 0.10   # std of internal noise ≈ this fraction × Q_int
GAIN_NOISE_INT_ABS = 0.12   # kW, additive floor (occupancy / plug variability)

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"]


# ═══════════════════════════════════════════════════════════════════════════
# Scenario generation
# ═══════════════════════════════════════════════════════════════════════════
def generate_scenario():
    """Synthetic winter-week scenario (cold-climate office building).

    Returns arrays of length ``N_STEPS + N_HORIZON`` so the MPC always has
    enough look-ahead data, even at the last simulation step.
    """
    n = N_STEPS + N_HORIZON
    t = np.arange(n) * DT            # time axis [hours]
    hod = t % 24                      # hour of day
    day_idx = np.minimum((t / 24).astype(int), 7)
    dow = day_idx % 7                 # day of week (0 = Mon)

    # ── Outdoor temperature (cold snap peaks Thursday) ─────────────────
    daily_means = np.array([-5, -4, -8, -12, -9, -3, -2, -3], dtype=float)
    day_centres = np.arange(8) * 24 + 12
    T_mean = np.interp(t, day_centres, daily_means)
    T_ext = T_mean + 4.0 * np.sin(2 * np.pi * (t - 3) / 24)

    # ── Solar gains (kW) ──────────────────────────────────────────────
    clearness = np.array([0.8, 0.9, 0.3, 0.4, 0.7, 1.0, 0.6, 0.7])
    Q_sol = np.zeros(n)
    for i in range(n):
        h = hod[i]
        if 8 <= h <= 16:
            Q_sol[i] = 2.5 * clearness[day_idx[i]] * np.sin(np.pi * (h - 8) / 8)

    # ── Internal gains (kW) ───────────────────────────────────────────
    Q_int = np.full(n, 0.3)          # standby equipment baseline
    for i in range(n):
        if dow[i] < 5 and 8 <= hod[i] < 18:
            h = hod[i]
            if h < 9:
                Q_int[i] = 1.5 + 1.5 * (h - 8)
            elif h >= 17:
                Q_int[i] = 1.5 + 1.5 * (18 - h)
            else:
                Q_int[i] = 3.0

    # ── Electricity prices ($/kWh) ────────────────────────────────────
    prices = np.full(n, 0.04)        # default off-peak
    for i in range(n):
        h = hod[i]
        if dow[i] < 5:               # weekday structure
            if 6 <= h < 9 or 16 <= h < 22:
                prices[i] = 0.10     # shoulder / evening peak
            elif 9 <= h < 16:
                prices[i] = 0.07     # midday
        if day_idx[i] in DR_DAYS and 16 <= h < 20:
            prices[i] = 0.60         # DR critical-peak event

    # ── Comfort bounds and reference temperature (°C) ───────────────
    T_min = np.full(n, 18.0)          # hard lower bound
    T_max = np.full(n, 25.0)         # hard upper bound (also unoccupied)
    T_ref = np.full(n, 18.0)          # setpoint reference
    for i in range(n):
        if dow[i] < 5 and 7 <= hod[i] < 18:
            T_min[i] = 20.0
            T_max[i] = 23.0
            T_ref[i] = 20.5

    return t, T_ext, Q_sol, Q_int, prices, T_min, T_max, T_ref


def perturb_plant_gains(Q_sol_nom, Q_int_nom, rng):
    """Gaussian perturbations on solar and internal gains for the **plant only**.

    The MPC optimisation uses ``Q_sol_nom``, ``Q_int_nom`` in its disturbance
    forecast; ``simulate_step`` uses the returned **actual** series so model
    mismatch reflects imperfect forecasts.

    Parameters
    ----------
    Q_sol_nom, Q_int_nom : ndarray
        Nominal profiles from :func:`generate_scenario` (same length).
    rng : numpy.random.Generator

    Returns
    -------
    Q_sol_act, Q_int_act : ndarray
        Non-negative perturbed gains (kW).
    """
    scale_sol = GAIN_NOISE_SOL_REL * (np.abs(Q_sol_nom) + 0.15) + GAIN_NOISE_SOL_ABS
    Q_sol_act = Q_sol_nom + rng.normal(0.0, scale_sol, size=Q_sol_nom.shape)
    Q_sol_act = np.clip(Q_sol_act, 0.0, None)

    scale_int = GAIN_NOISE_INT_REL * np.maximum(Q_int_nom, 0.2) + GAIN_NOISE_INT_ABS
    Q_int_act = Q_int_nom + rng.normal(0.0, scale_int, size=Q_int_nom.shape)
    Q_int_act = np.clip(Q_int_act, 0.05, None)

    return Q_sol_act, Q_int_act


# ═══════════════════════════════════════════════════════════════════════════
# Simulation loops
# ═══════════════════════════════════════════════════════════════════════════
def run_mpc_simulation(Ad, Bud, Bdd, scenario, params, Q_sol_act, Q_int_act):
    """Run MPC over the full week, returning state & control trajectories.

    Forecast disturbances use nominal ``Q_sol``, ``Q_int`` from ``scenario``;
    the building model steps with ``Q_sol_act``, ``Q_int_act``.
    """
    t, T_ext, Q_sol, Q_int, prices, T_min, T_max, T_ref = scenario
    mpc = BuildingMPC(Ad, Bud, Bdd, params["Q_max"], DT, N_HORIZON)

    x = np.zeros((N_STEPS + 1, 2))
    x[0] = [18.0, 18.0]               # weekend setback initial state
    Q_heat = np.zeros(N_STEPS)
    thursday_replans = []             # (step_index k, Q_plan ndarray) for replot

    t0 = time.time()
    print("Running MPC simulation …")
    for k in range(N_STEPS):
        if k % 96 == 0:
            print(f"  Day {k // 96 + 1}/7")

        sl = slice(k, k + N_HORIZON)
        d_forecast = np.column_stack([T_ext[sl], Q_sol[sl], Q_int[sl]])

        q, Q_plan, _, _ = mpc.solve(
            x[k], d_forecast, prices[sl], T_min[sl], T_max[sl], T_ref[sl]
        )
        Q_heat[k] = np.clip(q, 0, params["Q_max"])

        if int((k * DT) // 24) == DR_DETAIL_DAY:
            thursday_replans.append((k, np.asarray(Q_plan, dtype=float).copy()))

        d_k = np.array([T_ext[k], Q_sol_act[k], Q_int_act[k]])
        x[k + 1] = simulate_step(x[k], Q_heat[k], d_k, Ad, Bud, Bdd)

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f} s, {elapsed/N_STEPS*1000:.1f} ms/step)\n")
    return x, Q_heat, thursday_replans


def run_baseline_simulation(Ad, Bud, Bdd, scenario, params, Q_sol_act, Q_int_act):
    """PI thermostat with occupancy schedule (price-unaware).

    Mimics a well-tuned BMS: occupancy-based setpoints with a 1-hour
    pre-heat ramp before morning occupancy and night setback.
    Integral action eliminates steady-state offset.
    Uses the same perturbed gains as the MPC plant for a fair comparison.
    """
    t, T_ext, Q_sol, Q_int, prices, T_min, T_max, T_ref = scenario
    Q_max = params["Q_max"]
    K_p = 4.0     # proportional gain  [kW/K]
    K_i = 1.5     # integral gain      [kW/(K·h)]

    x = np.zeros((N_STEPS + 1, 2))
    x[0] = [18.0, 18.0]
    Q_heat = np.zeros(N_STEPS)
    e_int = 0.0   # integral of error [K·h]

    print("Running baseline (PI thermostat) simulation …")
    for k in range(N_STEPS):
        T_i = x[k, 0]
        h = (k * DT) % 24
        dow = int(k * DT / 24) % 7

        T_sp = T_ref[k]                        # 21 °C occupied, 18 °C otherwise
        if dow < 5 and 6 <= h < 7:             # pre-heat ramp (1 h before)
            T_sp = 21.0

        err = T_sp - T_i
        e_int += err * DT
        e_int = np.clip(e_int, 0, Q_max / K_i)  # anti-windup (heat only)

        Q_heat[k] = np.clip(K_p * err + K_i * e_int, 0, Q_max)

        d_k = np.array([T_ext[k], Q_sol_act[k], Q_int_act[k]])
        x[k + 1] = simulate_step(x[k], Q_heat[k], d_k, Ad, Bud, Bdd)

    print("  Done.\n")
    return x, Q_heat


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════
def compute_metrics(Q_heat, x, prices, T_min, T_max):
    """Compute energy, cost, comfort, and peak-demand metrics."""
    n = len(Q_heat)
    energy = np.sum(Q_heat) * DT                          # kWh
    cost = np.sum(Q_heat * prices[:n]) * DT                # $
    peak = np.max(Q_heat)                                  # kW

    T_i = x[1:, 0]
    viol_lo = np.maximum(T_min[:n] - T_i, 0)
    viol_hi = np.maximum(T_i - T_max[:n], 0)

    occ = T_min[:n] > 18.5        # occupied-hour mask (T_min = 20 when occupied)
    occ_viol = np.sum((viol_lo[occ] + viol_hi[occ]) > 0.1) * DT
    mean_Ti_occ = np.mean(T_i[occ]) if occ.any() else np.nan

    return dict(energy=energy, cost=cost, peak=peak,
                occ_viol_hours=occ_viol, mean_Ti_occ=mean_Ti_occ)


def print_summary(m_mpc, m_base):
    """Pretty-print the comparison table and analysis."""
    de = (m_base["energy"] - m_mpc["energy"]) / m_base["energy"] * 100
    dc = (m_base["cost"] - m_mpc["cost"]) / m_base["cost"] * 100
    dp = (m_base["peak"] - m_mpc["peak"]) / m_base["peak"] * 100
    avg_mpc  = m_mpc["cost"] / m_mpc["energy"] if m_mpc["energy"] > 0 else 0
    avg_base = m_base["cost"] / m_base["energy"] if m_base["energy"] > 0 else 0

    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║              Weekly Simulation Summary                       ║")
    print("╠═══════════════════════════════════════════════════════════════╣")
    print(f"║  {'Metric':<28s} {'MPC':>10s} {'Baseline':>10s} {'Δ':>8s} ║")
    print("╠═══════════════════════════════════════════════════════════════╣")
    print(f"║  {'Total energy (kWh)':<28s} {m_mpc['energy']:>10.1f} {m_base['energy']:>10.1f} {de:>+7.1f}% ║")
    print(f"║  {'Total cost ($)':<28s} {m_mpc['cost']:>10.2f} {m_base['cost']:>10.2f} {dc:>+7.1f}% ║")
    print(f"║  {'Avg cost ($/kWh)':<28s} {avg_mpc:>10.4f} {avg_base:>10.4f} {'':>8s} ║")
    print(f"║  {'Peak demand (kW)':<28s} {m_mpc['peak']:>10.1f} {m_base['peak']:>10.1f} {dp:>+7.1f}% ║")
    print(f"║  {'Comfort viol. occupied (h)':<28s} {m_mpc['occ_viol_hours']:>10.1f} {m_base['occ_viol_hours']:>10.1f} {'':>8s} ║")
    print(f"║  {'Mean T_i occupied (°C)':<28s} {m_mpc['mean_Ti_occ']:>10.1f} {m_base['mean_Ti_occ']:>10.1f} {'':>8s} ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print("Analysis:")
    print(f"  • MPC achieves {dc:.1f}% cost savings (avg {avg_mpc:.3f} vs {avg_base:.3f} $/kWh).")
    if de > 0:
        print(f"  • MPC uses {de:.1f}% less heating energy than the baseline this week.")
    elif de < 0:
        print(f"  • MPC uses {-de:.1f}% more heating energy (e.g. pre-heat / forecast mismatch).")
    else:
        print(f"  • Total heating energy is essentially the same for both controllers.")
    if m_mpc["occ_viol_hours"] < 0.05 and m_base["occ_viol_hours"] < 0.05:
        print("  • No meaningful occupied-hour comfort-band violations for either controller.")
    else:
        print(
            f"  • Occupied-hour comfort violations: MPC {m_mpc['occ_viol_hours']:.1f} h, "
            f"baseline {m_base['occ_viol_hours']:.1f} h (MPC can slip when actual gains exceed forecast)."
        )
    print("  • During DR windows, MPC tends to cut heating during $0.60/kWh peaks after pre-charging mass.")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 64)
    print("  Building Heating MPC – Demand Response Simulation")
    print("=" * 64 + "\n")

    params = DEFAULT_PARAMS
    print_model_info(params)

    # Discretise the thermal model
    Ad, Bud, Bdd = discretize(DT, params)

    # Generate scenario (nominal = MPC forecast); plant sees perturbed gains
    scenario = generate_scenario()
    t_sim = np.arange(N_STEPS) * DT
    _, _, Q_sol_nom, Q_int_nom, *_ = scenario
    rng = np.random.default_rng(RNG_SEED)
    Q_sol_act, Q_int_act = perturb_plant_gains(Q_sol_nom, Q_int_nom, rng)
    rms_sol = float(np.sqrt(np.mean((Q_sol_act[:N_STEPS] - Q_sol_nom[:N_STEPS]) ** 2)))
    rms_int = float(np.sqrt(np.mean((Q_int_act[:N_STEPS] - Q_int_nom[:N_STEPS]) ** 2)))
    print(
        f"Plant gain noise (vs MPC forecast): RMS(Q_sol) = {rms_sol:.3f} kW, "
        f"RMS(Q_int) = {rms_int:.3f} kW  (seed={RNG_SEED})\n"
    )

    # Run both controllers (same perturbed plant for fair comparison)
    x_mpc, Q_mpc, thursday_replans = run_mpc_simulation(
        Ad, Bud, Bdd, scenario, params, Q_sol_act, Q_int_act
    )
    x_base, Q_base = run_baseline_simulation(Ad, Bud, Bdd, scenario, params, Q_sol_act, Q_int_act)

    # Compute metrics (indices: 4=prices, 5=T_min, 6=T_max)
    m_mpc  = compute_metrics(Q_mpc, x_mpc, scenario[4], scenario[5], scenario[6])
    m_base = compute_metrics(Q_base, x_base, scenario[4], scenario[5], scenario[6])
    print_summary(m_mpc, m_base)

    # Generate plots
    fig1 = utils.plot_mpc_overview(
        t_sim, scenario, x_mpc, Q_mpc,
        n_steps=N_STEPS, n_days=N_DAYS, day_names=DAY_NAMES, dr_days=DR_DAYS,
        q_sol_act=Q_sol_act,
    )
    fig1.savefig("fig_mpc_overview.png", dpi=150, bbox_inches="tight")
    print("Saved fig_mpc_overview.png")

    fig2 = utils.plot_comparison(
        t_sim, scenario, x_mpc, Q_mpc, x_base, Q_base, m_mpc, m_base,
        n_steps=N_STEPS, n_days=N_DAYS, day_names=DAY_NAMES, dr_days=DR_DAYS, dt=DT,
    )
    fig2.savefig("fig_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved fig_comparison.png")

    fig3 = utils.plot_dr_day(
        t_sim, scenario, x_mpc, Q_mpc, x_base, Q_base,
        n_steps=N_STEPS, dr_detail_day=DR_DETAIL_DAY,
    )
    fig3.savefig("fig_dr_day_detail.png", dpi=150, bbox_inches="tight")
    print("Saved fig_dr_day_detail.png")

    fig4 = utils.plot_mpc_replan_thursday(
        scenario, thursday_replans, Q_mpc, dt=DT, dr_detail_day=DR_DETAIL_DAY,
    )
    fig4.savefig("fig_mpc_replan_thursday.png", dpi=150, bbox_inches="tight")
    print("Saved fig_mpc_replan_thursday.png")

    plt.show()


if __name__ == "__main__":
    main()
