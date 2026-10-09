"""MIMO PID tuning via iterated LMI restriction.

Python port of the algorithm in

    S. Boyd, M. Hast, and K. J. Astrom, "MIMO PID tuning via iterated LMI
    restriction", Int. J. Robust & Nonlinear Control, 26:1718-1731, 2016.
    https://web.stanford.edu/~boyd/papers/pdf/mimo_pid_tuning.pdf

and of Daniel Rubin's MATLAB implementation (YALMIP/SeDuMi):
    https://github.com/rubindan/mimoPIDtune

Problem solved (paper eq. (2), sampled at N frequencies w_k):

    minimize    ||(P(0) K_I)^{-1}||
    subject to  ||S(j w_k)|| <= Smax_k,  ||T(j w_k)|| <= Tmax_k,  ||Q(j w_k)|| <= Qmax_k

with C(s) = K_P + K_I / s + K_D s / (1 + tau s),  S = (I + P C)^{-1},
T = P C S,  Q = C S.  Each constraint is a quadratic matrix inequality
Z^* Z >= Y^* Y that is replaced by its LMI restriction (paper eq. (5))

    [[ Z^* Z0 + Z0^* Z - Z0^* Z0,  Y^* ],
     [ Y,                          I   ]]  >= 0

linearised at the current iterate Z0; the resulting SDP is solved repeatedly
until the objective stops improving (paper Section 6).
"""

from __future__ import annotations

import time
from typing import Callable, Union

import numpy as np
import cvxpy as cp

PlantLike = Union[Callable[[float], np.ndarray], object]


# ---------------------------------------------------------------------------
# Plant helpers
# ---------------------------------------------------------------------------
def fotd_plant(K, T, L):
    """Return P(w) -> P(jw) for a matrix of first-order-plus-time-delay elements.

    P_ij(s) = K_ij exp(-L_ij s) / (T_ij s + 1).

    The delays are evaluated exactly (no Pade approximation), which is all the
    tuning algorithm needs since it only ever samples P on a frequency grid.
    """
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    L = np.asarray(L, dtype=float)

    def P(w: float) -> np.ndarray:
        s = 1j * w
        return K * np.exp(-L * s) / (T * s + 1.0)

    P.shape = K.shape
    return P


def _as_freqresp_fn(P: PlantLike) -> Callable[[float], np.ndarray]:
    """Accept a callable P(w) or a python-control LTI/FRD and return P(w)."""
    if callable(P) and not hasattr(P, "frequency_response"):
        return P
    from control import frequency_response  # local import: optional dependency

    def Pfn(w: float) -> np.ndarray:
        resp = frequency_response(P, [max(w, 0.0)]).response  # (p, m, 1)
        return np.asarray(resp)[:, :, 0]

    return Pfn


# ---------------------------------------------------------------------------
# LMI restriction (paper eq. (5))
# ---------------------------------------------------------------------------
def make_lmi(Z, Z0: np.ndarray, Y):
    """Hermitian block matrix whose PSD-ness implies Z^* Z >= Y^* Y.

    Z, Y: cvxpy affine expressions (possibly complex); Z0: numeric linearisation
    point.  Note the *conjugate* transpose: the blocks are complex at every
    frequency, so Z.T would be wrong.
    """
    Z0 = np.asarray(Z0)
    p = Y.shape[0]
    Yh = Y.H if isinstance(Y, cp.Expression) else np.asarray(Y).conj().T
    top_left = Z.H @ Z0 + Z0.conj().T @ Z - Z0.conj().T @ Z0
    return cp.bmat([[top_left, Yh], [Y, np.eye(p)]])


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def auto_mimo_pid(P, w, Smax, Tmax, Qmax, tau, options=None, *, solver="CLARABEL",
                  solver_opts=None, verbose=True):
    """Autotune a MIMO PID controller for a stable plant by iterated LMI restriction.

    Parameters
    ----------
    P : callable or python-control LTI (or a list of them)
        Nominal plant.  A callable must map a frequency ``w`` (rad/s) to the
        complex ``(p, m)`` matrix ``P(jw)``; see :func:`fotd_plant`.  Pass a
        list ``[P_nominal, P_1, ...]`` to enforce the constraints for several
        plant cases (robust design); ``P[0]`` is used for the objective.
    w : array, shape (N,)
        Frequency grid (rad/s) on which the H-infinity constraints are sampled.
    Smax, Tmax, Qmax : float or array, shape (N,)
        Upper bounds on ||S||, ||T||, ||Q|| at each frequency.  Reasonable
        values are 1.1-1.6 for Smax/Tmax and 3-10 x 1/sigma_min(P(0)) for Qmax.
    tau : float
        Derivative filter time constant.
    options : dict, optional
        ``initial``   : dict with ``Kp``, ``Ki``, ``Kd`` arrays to start from.
        ``epsilon``   : gain of the default initial controller
                        ``Ki = epsilon * pinv(P(0))`` (default 0.01, as in the paper).
        ``structure`` : dict with 0/1 masks ``Kp``, ``Ki``, ``Kd``; zeros are
                        forced to zero (e.g. diagonal PID).
        ``max_iter``  : maximum number of SDP iterations (default 10).
        ``tol``       : stop when the objective improves by less than ``tol``
                        (default 1e-3, absolute, as in the MATLAB code).
    solver : str
        Any cvxpy SDP-capable solver ("CLARABEL", "SCS", "MOSEK", ...).
    solver_opts : dict, optional
        Extra keyword arguments passed to ``Problem.solve``.

    Returns
    -------
    result : dict with keys
        ``Kp``, ``Ki``, ``Kd`` : optimised gain matrices (m, p)
        ``objval``             : ||(P(0) Ki)^{-1}||
        ``t``                  : final SDP objective (sigma_min(P(0) Ki) lower bound)
        ``iterations``         : number of SDP solves
        ``history``            : objective value after every iteration
        ``controller``         : callable C(w) -> C(jw)
    """
    options = dict(options or {})
    plants = P if isinstance(P, (list, tuple)) else [P]
    plants = [_as_freqresp_fn(Pi) for Pi in plants]

    w = np.asarray(w, dtype=float)
    N = len(w)
    Smax = np.broadcast_to(np.asarray(Smax, dtype=float), (N,))
    Tmax = np.broadcast_to(np.asarray(Tmax, dtype=float), (N,))
    Qmax = np.broadcast_to(np.asarray(Qmax, dtype=float), (N,))

    # DC gains and frequency responses, computed once.
    P0 = [np.real(Pi(0.0)) for Pi in plants]
    Pk = [[Pi(wk) for wk in w] for Pi in plants]
    p, m = P0[0].shape  # p outputs, m inputs
    Ip = np.eye(p)

    # --- initial controller (paper Section 6.1) ----------------------------
    if "initial" in options:
        Kp0 = np.asarray(options["initial"]["Kp"], dtype=float)
        Ki0 = np.asarray(options["initial"]["Ki"], dtype=float)
        Kd0 = np.asarray(options["initial"]["Kd"], dtype=float)
    else:
        eps = options.get("epsilon", 0.01)
        Kp0 = np.zeros((m, p))
        Ki0 = eps * np.linalg.pinv(P0[0])
        Kd0 = np.zeros((m, p))

    structure = options.get("structure")
    max_iter = options.get("max_iter", 10)
    tol = options.get("tol", 1e-3)

    def controller_at(Kp, Ki, Kd, wk):
        s = 1j * wk
        return Kp + (1.0 / s) * Ki + (s / (1.0 + tau * s)) * Kd

    if verbose:
        print(f"Optimising a {m}x{p} MIMO PID controller "
              f"({N} frequencies, {len(plants)} plant case(s), solver={solver})")
    tic = time.time()

    solve_kwargs = dict(solver_opts or {})
    if solver.upper() == "CLARABEL" and not solve_kwargs:
        # Clarabel's default 1e-8 tolerances are not reachable on these SDPs and
        # only trigger "optimal_inaccurate"; 1e-7 is far below the 1e-3 stopping
        # tolerance of the outer iteration.
        solve_kwargs = dict(tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-7)

    t_prev = 0.0
    t_val = 0.0
    history = []
    it = 0
    for it in range(1, max_iter + 1):
        t = cp.Variable()
        Kp = cp.Variable((m, p))
        Ki = cp.Variable((m, p))
        Kd = cp.Variable((m, p))

        constraints = []
        if structure is not None:
            for var, key in ((Kp, "Kp"), (Ki, "Ki"), (Kd, "Kd")):
                mask = np.asarray(structure[key]) == 0
                if mask.any():
                    constraints.append(var[mask] == 0)

        # Objective: maximise t with sigma_min(P(0) Ki) >= t,
        # i.e. the QMI (P0 Ki)^* (P0 Ki) >= t^2 I, restricted at P0 Ki0.
        constraints.append(make_lmi(P0[0] @ Ki, P0[0] @ Ki0, t * Ip) >> 0)

        for icase in range(len(plants)):
            for k in range(N):
                wk = w[k]
                Pkw = Pk[icase][k]
                Ck = controller_at(Kp, Ki, Kd, wk)
                Ck0 = controller_at(Kp0, Ki0, Kd0, wk)
                Z = Ip + Pkw @ Ck
                Z0 = Ip + Pkw @ Ck0
                # ||S_k|| <= Smax   <=>  Z^*Z >= (1/Smax)^2 I
                constraints.append(make_lmi(Z, Z0, (1.0 / Smax[k]) * Ip) >> 0)
                # ||T_k|| <= Tmax   <=>  Z^*Z >= (1/Tmax)^2 (P C)^*(P C)
                constraints.append(make_lmi(Z, Z0, (1.0 / Tmax[k]) * (Pkw @ Ck)) >> 0)
                # ||Q_k|| <= Qmax   <=>  Z^*Z >= (1/Qmax)^2 C^* C
                constraints.append(make_lmi(Z, Z0, (1.0 / Qmax[k]) * Ck) >> 0)

        prob = cp.Problem(cp.Maximize(t), constraints)
        prob.solve(solver=solver, **solve_kwargs)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"SDP failed at iteration {it}: status {prob.status}")

        Kp0, Ki0, Kd0 = Kp.value, Ki.value, Kd.value
        t_val = float(t.value)
        obj = np.linalg.norm(np.linalg.inv(P0[0] @ Ki0), 2)
        history.append(obj)
        if verbose:
            print(f"  iter {it:2d}: t = {t_val:.5f}   ||(P(0)Ki)^-1|| = {obj:.4f}   "
                  f"[{prob.status}, {time.time() - tic:.1f}s]")

        if abs(t_val - t_prev) < tol:
            break
        t_prev = t_val

    objval = float(np.linalg.norm(np.linalg.inv(P0[0] @ Ki0), 2))
    if verbose:
        print(f"Converged after {it} iterations in {time.time() - tic:.1f}s; "
              f"||(P(0)Ki)^-1|| = {objval:.4f}")

    def C(wk: float) -> np.ndarray:
        return controller_at(Kp0, Ki0, Kd0, wk)

    return {
        "Kp": Kp0, "Ki": Ki0, "Kd": Kd0,
        "objval": objval, "t": t_val, "iterations": it, "history": history,
        "controller": C, "tau": tau,
    }


# ---------------------------------------------------------------------------
# Closed-loop analysis helpers
# ---------------------------------------------------------------------------
def closed_loop_sigmas(P, C, w):
    """Singular values of S, T, Q = (I+PC)^-1, PC S, C S on the grid ``w``.

    Returns three arrays of shape (N, p) sorted in decreasing order.
    """
    Pfn = _as_freqresp_fn(P)
    S_sv, T_sv, Q_sv = [], [], []
    for wk in w:
        Pk, Ck = Pfn(wk), C(wk)
        Ip = np.eye(Pk.shape[0])
        S = np.linalg.inv(Ip + Pk @ Ck)
        S_sv.append(np.linalg.svd(S, compute_uv=False))
        T_sv.append(np.linalg.svd(Pk @ Ck @ S, compute_uv=False))
        Q_sv.append(np.linalg.svd(Ck @ S, compute_uv=False))
    return np.array(S_sv), np.array(T_sv), np.array(Q_sv)


def mimo_ss_from_entries(entries):
    """Stack a nested list of SISO python-control systems into one MIMO StateSpace.

    ``entries[i][j]`` is the SISO transfer function from input j to output i.
    Only SISO tf->ss conversions are needed, so this works without Slycot.
    """
    from control import ss
    from scipy.linalg import block_diag

    p, m = len(entries), len(entries[0])
    blocks = [[ss(entries[i][j]) for j in range(m)] for i in range(p)]
    A = block_diag(*[blocks[i][j].A for i in range(p) for j in range(m)])
    n = A.shape[0]
    B = np.zeros((n, m))
    C = np.zeros((p, n))
    D = np.zeros((p, m))
    k = 0
    for i in range(p):
        for j in range(m):
            blk = blocks[i][j]
            ni = blk.A.shape[0]
            B[k:k + ni, j] = blk.B[:, 0]
            C[i, k:k + ni] = blk.C[0, :]
            D[i, j] = blk.D[0, 0]
            k += ni
    return ss(A, B, C, D)


def pid_state_space(Kp, Ki, Kd, tau):
    """python-control StateSpace of C(s) = Kp + Ki/s + Kd s/(1+tau s) (MIMO)."""
    from control import tf

    Kp, Ki, Kd = (np.asarray(K, dtype=float) for K in (Kp, Ki, Kd))
    m, p = Kp.shape
    entries = [[None] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            # (Kp s (1+tau s) + Ki (1+tau s) + Kd s^2) / (s (1+tau s))
            entries[i][j] = tf([Kp[i, j] * tau + Kd[i, j], Kp[i, j] + Ki[i, j] * tau, Ki[i, j]],
                               [tau, 1.0, 0.0])
    return mimo_ss_from_entries(entries)
