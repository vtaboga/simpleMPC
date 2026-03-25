# Linear MPC for building heating (R2C2)

**Toy teaching environment.** The goal is to learn MPC on a small, readable stack. The **R2C2 thermal model** and **scenario data** (outdoor temperature, solar curve, internal gains, prices) are **simple heuristics**, not calibrated or weather-file accurate. In my view this is the **minimum** that still feels physically plausible for classroom experiments.

A **single-zone** RC model is controlled by **receding-horizon linear MPC** over one winter week with **time-of-use prices** and a **demand-response spike**. Heating is one bounded power input (no detailed HVAC plant).

---

## Layout

| File | Role |
|------|------|
| `building_model.py` | R2C2: continuous-time matrices, ZOH discretisation, one-step simulation |
| `mpc_controller.py` | MPC: CVXPY **LP** (energy cost + comfort slack), **CLARABEL**, parametric problem, warm start |
| `run_simulation.py` | Synthetic week, MPC vs PI baseline, metrics |
| `utils.py` | Matplotlib figures |
| `requirements.txt` | Dependencies |

---

## MPC (current formulation)

There is **no setpoint-tracking term** in the objective anymore: the optimiser only minimises **predicted electricity cost** plus a **large penalty on comfort slacks** so soft bounds stay tight. Dynamics and soft comfort constraints are linear, so the problem is a **linear program (LP)**.

Comfort bands and a `T_ref` series are still passed into the controller for a consistent API; **`T_ref` does not appear in the cost**—only `T_min`, `T_max`, prices, and slacks matter for the optimum.

The **baseline** is a price-unaware **PI thermostat** following the scenario setpoints (with a short pre-heat). The plant can use **noisy** solar/internal gains while the MPC forecast uses the **nominal** profiles (`RNG_SEED` in `run_simulation.py`).

---

## Run

```bash
cd linear_mpc
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
python3 run_simulation.py
```

Saves `fig_mpc_overview.png`, `fig_comparison.png`, `fig_dr_day_detail.png`, `fig_mpc_replan_thursday.png` (and may open a plot window depending on your matplotlib backend).

---

## Example results (defaults, `RNG_SEED=42`)

Representative numbers from one run; small changes are normal across machines / solver builds.

| Metric | MPC | Baseline |
|--------|-----|----------|
| Heating energy (kWh) | ~497 | ~494 |
| Electricity cost ($) | ~36.9 | ~42.9 |
| Avg. $/kWh | ~0.074 | ~0.087 |
| Peak heating (kW) | 15.0 | 15.0 |
| Comfort violations, occupied (h)* | ~5.8 | ~0 |
| Mean $T_i$ when occupied (°C) | ~20.7 | ~20.6 |

\*Hours with $|T_i - [T_{\min},T_{\max}]| > 0.1$ °C when occupied (weekdays 07:00–18:00, bands 20–23 °C).

**Takeaway:** MPC still **shifts load toward cheap periods** (lower bill) with **similar total energy**, but without a tracking penalty it may **ride the lower comfort bound** more often; combined with **forecast mismatch** on gains, occupied-band violations can appear while the baseline PI stays tighter to setpoint.

---

## Teaching use

Code is for coursework and self-study. Cite Zanetti et al. (2025) if you reference building MPC / RC modelling in reports.
