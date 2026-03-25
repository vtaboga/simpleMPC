"""
R2C2 Thermal Model of a Building Zone

Simplified resistance-capacitance (RC) thermal network for a single building
zone.  Two thermal resistances and two capacitances capture both the fast
 response of indoor air and the slow thermal storage in the building envelope.

Model Structure


    T_ext ---[ R_we ]--- T_w ---[ R_iw ]--- T_i
                          |                   |
                         C_w                 C_i
                          |                   |
                         GND                 GND

States
    T_i  : indoor air temperature  [°C]
    T_w  : envelope (wall/slab) temperature  [°C]

Control input
    Q_heat : HVAC heating power delivered to the zone  [kW]

Disturbances
    T_ext  : outdoor air temperature  [°C]
    Q_sol  : solar heat gains through glazing  [kW]
    Q_int  : internal gains (occupants, equipment, lighting)  [kW]

Continuous-time state-space form
    dx/dt = A·x + B_u·u + B_d·d

Units throughout: kW, °C, hours, kWh/K, K/kW.

"""

import numpy as np
from scipy.linalg import expm

# ── Default parameters: 200 m² building, heavy concrete, cold climate ───────
DEFAULT_PARAMS = dict(
    C_i=0.8,        # Interior thermal capacitance  [kWh/K]
    C_w=12.0,       # Envelope thermal capacitance   [kWh/K]
    R_iw=1.0,       # Air ↔ wall inner surface       [K/kW]
    R_we=4.5,       # Wall centre ↔ exterior          [K/kW]
    alpha_i=0.4,    # Fraction of solar gains → air node
    alpha_w=0.6,    # Fraction of solar gains → wall node
    Q_max=15.0,     # Maximum heating power           [kW]
)


def get_continuous_ss(params=None):
    """Return continuous-time matrices (A, B_u, B_d).

    x = [T_i, T_w]^T    u = [Q_heat]    d = [T_ext, Q_sol, Q_int]^T
    """
    p = params or DEFAULT_PARAMS
    Ci, Cw = p["C_i"], p["C_w"]
    Riw, Rwe = p["R_iw"], p["R_we"]
    ai, aw = p["alpha_i"], p["alpha_w"]

    A = np.array([
        [-1 / (Ci * Riw),                1 / (Ci * Riw)],
        [ 1 / (Cw * Riw),  -1 / (Cw * Riw) - 1 / (Cw * Rwe)],
    ])
    B_u = np.array([
        [1 / Ci],
        [0.0],
    ])
    B_d = np.array([
        [0.0,             ai / Ci,  1 / Ci],
        [1 / (Cw * Rwe),  aw / Cw,  0.0],
    ])
    return A, B_u, B_d


def discretize(dt, params=None):
    """Exact zero-order-hold discretization via the matrix exponential.

    Parameters
    ----------
    dt : float
        Time step [hours].

    Returns
    -------
    Ad, Bud, Bdd : ndarray
        Discrete-time state-space matrices.
    """
    A, Bu, Bd = get_continuous_ss(params)
    n = A.shape[0]
    m = Bu.shape[1] + Bd.shape[1]

    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * dt
    M[:n, n : n + Bu.shape[1]] = Bu * dt
    M[:n, n + Bu.shape[1] :] = Bd * dt

    eM = expm(M)
    Ad = eM[:n, :n]
    Bud = eM[:n, n : n + Bu.shape[1]]
    Bdd = eM[:n, n + Bu.shape[1] :]
    return Ad, Bud, Bdd


def simulate_step(x, u, d, Ad, Bud, Bdd):
    """Propagate one discrete time step.

    Parameters
    ----------
    x : ndarray (2,)  [T_i, T_w]
    u : float         heating power  [kW]
    d : ndarray (3,)  [T_ext, Q_sol, Q_int]
    """
    return Ad @ x + Bud.flatten() * u + Bdd @ d


def print_model_info(params=None):
    """Print key physical properties of the RC model."""
    p = params or DEFAULT_PARAMS
    A, _, _ = get_continuous_ss(p)
    eigvals = np.linalg.eigvals(A)
    taus = -1.0 / np.real(eigvals)

    ua = 1 / (p["R_iw"] + p["R_we"])
    print("── R2C2 Building Model Properties ──")
    print(f"  Overall UA value :  {ua:.3f} kW/K")
    print(f"  Heat loss at ΔT=30 K : {30 * ua:.1f} kW")
    print(f"  Fast time constant (air) :  {min(taus):.1f} h  ({min(taus)*60:.0f} min)")
    print(f"  Slow time constant (wall):  {max(taus):.1f} h  ({max(taus)/24:.1f} days)")
    print(f"  Max heating power : {p['Q_max']:.0f} kW\n")
