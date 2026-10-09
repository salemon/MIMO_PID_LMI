"""Reproduce Section 8 of Boyd, Hast & Astrom (2016) on the Wood-Berry column.

    python wood_berry_example.py              # full MIMO PID (paper Fig. 3, 5, 6)
    python wood_berry_example.py --diagonal   # diagonal PID (paper Fig. 4)
    python wood_berry_example.py -N 60        # coarser frequency grid, faster

Figures are written to ``figures/`` and shown unless ``--no-show`` is given.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

from auto_mimo_pid import (auto_mimo_pid, closed_loop_sigmas, fotd_plant,
                           mimo_ss_from_entries, pid_state_space)

# ---------------------------------------------------------------------------
# Wood-Berry binary distillation column (Wood & Berry, 1973)
#
#          [ 12.8 e^{-s}/(16.7 s + 1)   -18.9 e^{-3s}/(21.0 s + 1) ]
#   P(s) = [                                                       ]
#          [  6.6 e^{-7s}/(10.9 s + 1)  -19.4 e^{-3s}/(14.2 s + 1) ]
# ---------------------------------------------------------------------------
GAIN = [[12.8, -18.9], [6.6, -19.4]]
TIME_CONSTANT = [[16.7, 21.0], [10.9, 14.2]]
DELAY = [[1.0, 3.0], [7.0, 3.0]]

# Values reported in the paper (p. 1727 and p. 1729).
PAPER_MIMO = dict(
    Kp=np.array([[0.1750, -0.0470], [-0.0751, -0.0709]]),
    Ki=np.array([[0.0913, -0.0345], [0.0402, -0.0328]]),
    Kd=np.array([[0.1601, -0.0051], [0.0201, -0.1768]]),
    objval=2.25, iterations=7,
)
PAPER_DIAGONAL = dict(
    Kp=np.diag([0.1535, -0.0692]),
    Ki=np.diag([0.0210, -0.0136]),
    Kd=np.diag([0.1714, -0.1725]),
    objval=13.36, iterations=8,
)


def design(N=300, diagonal=False, solver="CLARABEL", verbose=True):
    P = fotd_plant(GAIN, TIME_CONSTANT, DELAY)
    w = np.logspace(-3, 3, N)
    P0 = P(0.0).real
    tau = 0.3
    Smax, Tmax = 1.4, 1.4
    Qmax = 3.0 / np.linalg.svd(P0, compute_uv=False).min()  # = 0.738

    options = {"epsilon": 0.01}
    if diagonal:
        # Paper: low-gain PI on the diagonal; the minus sign on (2,2) follows
        # the sign of P(0)_{22}.
        options["structure"] = {"Kp": np.eye(2), "Ki": np.eye(2), "Kd": np.eye(2)}
        options["initial"] = {"Kp": 1e-3 * np.diag([1, -1]),
                              "Ki": 1e-3 * np.diag([1, -1]),
                              "Kd": np.zeros((2, 2))}

    result = auto_mimo_pid(P, w, Smax, Tmax, Qmax, tau, options,
                           solver=solver, verbose=verbose)
    result.update(P=P, w=w, Smax=Smax, Tmax=Tmax, Qmax=Qmax)
    return result


def report(result, reference):
    np.set_printoptions(precision=4, suppress=True)
    print("\n=== Result vs. paper ===")
    for key in ("Kp", "Ki", "Kd"):
        print(f"{key}  (this code)\n{result[key]}")
        print(f"{key}  (paper)\n{reference[key]}")
    print(f"objective ||(P(0)Ki)^-1||: this code {result['objval']:.3f}, "
          f"paper {reference['objval']:.2f}")
    print(f"iterations: this code {result['iterations']}, paper {reference['iterations']}")

    wf = np.logspace(-3, 3, 3000)
    S, T, Q = closed_loop_sigmas(result["P"], result["controller"], wf)
    print(f"peak ||S|| = {S.max():.3f} (limit {result['Smax']}),  "
          f"peak ||T|| = {T.max():.3f} (limit {result['Tmax']}),  "
          f"peak ||Q|| = {Q.max():.3f} (limit {result['Qmax']:.3f})  on a 3000-point grid")


def plot_sigmas(result, outfile):
    import matplotlib.pyplot as plt

    w = np.logspace(-3, 3, 1000)
    S, T, Q = closed_loop_sigmas(result["P"], result["controller"], w)
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for ax, sv, bound, name in zip(axes, (S, T, Q),
                                   (result["Smax"], result["Tmax"], result["Qmax"]),
                                   ("S", "T", "Q")):
        ax.semilogx(w, 20 * np.log10(sv), "b")
        ax.axhline(20 * np.log10(bound), color="r", ls="--", label=f"{name}max")
        ax.set_ylabel(f"$\\sigma({name})$ [dB]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right")
    axes[0].set_title("Closed-loop singular values (cf. paper Fig. 3 / 4)")
    axes[-1].set_xlabel("frequency [rad/s]")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    return fig


def closed_loop_step(result, pade_order=6, t_end=100.0):
    """Step responses r -> y (T) and r -> u (Q) using a Pade approximation of the delays."""
    from control import feedback, pade, step_response, tf

    s = tf("s")
    entries = [[None] * 2 for _ in range(2)]
    for i in range(2):
        for j in range(2):
            pn, pd = pade(DELAY[i][j], pade_order)
            entries[i][j] = GAIN[i][j] / (TIME_CONSTANT[i][j] * s + 1) * tf(pn, pd)
    P = mimo_ss_from_entries(entries)
    C = pid_state_space(result["Kp"], result["Ki"], result["Kd"], result["tau"])
    T = feedback(P * C, np.eye(2))        # r -> y
    Q = feedback(C, P)                    # r -> u
    t = np.linspace(0, t_end, 1000)
    yT = step_response(T, t).outputs      # (2 outputs, 2 inputs, len(t))
    yQ = step_response(Q, t).outputs
    return t, yT, yQ


def plot_steps(result, outfile):
    import matplotlib.pyplot as plt

    t, yT, yQ = closed_loop_step(result)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for i in range(2):
        for j in range(2):
            axes[i, j].plot(t, yT[i, j], "b", label="$y$ (T)")
            axes[i, j].plot(t, yQ[i, j], "g", label="$u$ (Q)")
            axes[i, j].set_title(f"step on $r_{j + 1}$ -> $y_{i + 1}$, $u_{i + 1}$")
            axes[i, j].grid(alpha=0.3)
    axes[0, 0].legend()
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 1].set_xlabel("time [s]")
    fig.suptitle("Closed-loop step responses, 6th-order Pade delays (cf. paper Fig. 5 / 6)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-N", type=int, default=300, help="number of frequency samples (paper: 300)")
    ap.add_argument("--diagonal", action="store_true", help="design a diagonal (decentralised) PID")
    ap.add_argument("--solver", default="CLARABEL", help="cvxpy SDP solver (CLARABEL, SCS, ...)")
    ap.add_argument("--no-show", action="store_true", help="save figures without opening windows")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = design(N=args.N, diagonal=args.diagonal, solver=args.solver)
    report(result, PAPER_DIAGONAL if args.diagonal else PAPER_MIMO)

    os.makedirs(args.outdir, exist_ok=True)
    tag = "diagonal" if args.diagonal else "mimo"
    plot_sigmas(result, os.path.join(args.outdir, f"wood_berry_{tag}_sigma.png"))
    plot_steps(result, os.path.join(args.outdir, f"wood_berry_{tag}_step.png"))
    print(f"figures written to {args.outdir}/")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
