"""
Linear Model Predictive Controller for Building Heating
========================================================


              N-1
    min  J = Σ  [ price_k · Q_k · Δt + w_slack · (s⁻_k + s⁺_k) · Δt]
             k=0

    s.t.   x_{k+1}  = A_d x_k  + B_ud Q_k  + B_dd d_k    (dynamics)
           T_min_k − s⁻_k  ≤  T_i,k+1  ≤  T_max_k + s⁺_k (comfort)
           0 ≤ Q_k ≤ Q_max                                 (actuator)
           s⁻_k, s⁺_k ≥ 0                                  (slack)
           x_0 = x_measured                                 (initial)

"""

import numpy as np
import cvxpy as cp


class BuildingMPC:
    """MPC controller for building zone heating."""

    def __init__(self, Ad, Bud, Bdd, Q_max, dt, N_horizon,  w_track=0.001, w_slack=500.0):
        """
        Parameters
        ----------
        Ad, Bud, Bdd : ndarray   discrete state-space matrices
        Q_max        : float     max heating power [kW]
        dt           : float     time step [hours]
        N_horizon    : int       prediction horizon [steps]
        w_slack      : float     comfort-violation penalty [$/°C·h]
        """
        N = N_horizon
        nx = 2
        bud = Bud.flatten()

        # ── CVXPY parameters (updated before each solve) ──
        self.x0_par    = cp.Parameter(nx)
        self.dist_par  = cp.Parameter((N, 3))
        self.price_par = cp.Parameter(N, nonneg=True)
        self.tmin_par  = cp.Parameter(N)
        self.tmax_par  = cp.Parameter(N)
        self.tref_par  = cp.Parameter(N)          

        # ── Decision variables ──
        x    = cp.Variable((N + 1, nx), name="x")
        Q    = cp.Variable(N, name="Q")
        s_lo = cp.Variable(N, nonneg=True, name="s_lo")
        s_hi = cp.Variable(N, nonneg=True, name="s_hi")

        # ── Objective ──
        energy_cost     = self.price_par @ Q * dt
        comfort_penalty = w_slack * cp.sum(s_lo + s_hi) * dt

        objective = cp.Minimize(energy_cost + comfort_penalty)

        # ── Constraints ──
        cons = [x[0] == self.x0_par]
        for k in range(N):
            cons.append(
                x[k + 1] == Ad @ x[k] + bud * Q[k] + Bdd @ self.dist_par[k]
            )
            cons.append(x[k + 1, 0] >= self.tmin_par[k] - s_lo[k])
            cons.append(x[k + 1, 0] <= self.tmax_par[k] + s_hi[k])
            cons.append(Q[k] >= 0)
            cons.append(Q[k] <= Q_max)

        self.prob = cp.Problem(objective, cons)
        self._Q = Q
        self._x = x
        self.N = N

    # ──────────────────────────────────────────────────────────────────
    def solve(self, x0, disturbances, prices, T_min, T_max, T_ref):
        """Solve the MPC and return the first-step optimal heating power.

        Parameters
        ----------
        x0           : ndarray (2,)    – current state [T_i, T_w]
        disturbances : ndarray (N, 3)  – forecasted [T_ext, Q_sol, Q_int]
        prices       : ndarray (N,)    – electricity price forecast [$/kWh]
        T_min, T_max : ndarray (N,)    – comfort bounds [°C]
        T_ref        : ndarray (N,)    – reference temperature [°C]

        Returns
        -------
        q_first  : float     – optimal heating power for current step [kW]
        Q_plan   : ndarray   – full planned heating trajectory [kW]
        T_i_plan : ndarray   – predicted indoor-temperature trajectory [°C]
        cost     : float     – optimal objective value [$]
        """
        self.x0_par.value    = x0
        self.dist_par.value  = disturbances
        self.price_par.value = prices
        self.tmin_par.value  = T_min
        self.tmax_par.value  = T_max
        self.tref_par.value  = T_ref

        self.prob.solve(solver=cp.CLARABEL, warm_start=True)

        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return 0.0, np.zeros(self.N), np.full(self.N + 1, x0[0]), np.inf

        return (
            float(self._Q.value[0]),
            self._Q.value,
            self._x.value[:, 0],
            self.prob.value,
        )
