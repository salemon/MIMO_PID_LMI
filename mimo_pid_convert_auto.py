import numpy as np
import scipy.linalg as la
import control
from scipy.optimize import minimize

def auto_mimo_pid_convert(P, w, Smax, Tmax, Qmax, tau, Options=None):
    """
    AUTO_MIMO_PID Autotune a MIMO PID for a nominal plant by LMIs.
    [ S. Boyd, M. Hast, and J. Astrom, "MIMO PID tuning via iterated LMI ]
    [    restriction", Inter. Jur. of Robust & Nonlinear Control, 2016.   ]

    Usage:
    [ C, objval] = AUTO_MIMO_PID( P, w, Smax, Tmax, Qmax, tau, Options ) 
    returns the optimized MIMO PID controller C. Can also return objval,  
    the value of the objective after optimization.

    Inputs:
    P           plant in ss, tf, zpk, or frd form; input an array of plants 
                to account for uncertainty and variations in parameters
    w           vector of N frequencies responsibly chosen to cover the 
                dynamic range 
    Smax        sensitivity specification upper bound
    Tmax        low frequency sensitivity specification upper bound
    Qmax        cost of feedback specification upper bound
    tau         derivative action time constant 
    Options     optional dict which specifies additional options listed
                below

    Additional Options:
    Structure   substructure with fields Kp, Ki, and Kd; each subfield is a 
                matrix with 0/1 elements, where 0 constrain the corresponding 
                element to be 0.
    Initial,    substructure with fields Kp, Ki, and Kd; each subfield is a 
                matrix for initialization.
    Sign        substructure with fields Kp, Ki, and Kd; each subfield is an 
                integer: 1 for positive, -1 for negative
    maxInterp   maximal number of interpolations (default=10)

    Outputs: 
    C           optimized MIMO PID controller transfer matrix
    objval      value of the objective after optimization.   
    """
    
    def makeLMI(Z, Z0, Y):
        """Constructs the LMI"""
        return np.block([[Z.T @ Z0 + Z0.T @ Z - Z0.T @ Z0, Y.T],
                         [Y, np.eye(p)]])

    def sparsesdpvar(S):
        """Define sparse sdpvar with given structure S"""
        return np.where(S != 0, np.random.rand(*S.shape), 0)

    p, m = P(0).shape  # Number of inputs and outputs
    l = p*m  # Number of plants
    N = len(w)
    Pw0 = control.freqresp(P, w[0])[0]
    P0 = np.abs(Pw0) * np.sign(np.real(Pw0))

    if Options is None:
        Options = {}

    # Initialization of Kp, Ki, Kd:
    if 'Initial' in Options:
        Kp0 = Options['Initial']['Kp']
        Ki0 = Options['Initial']['Ki']
        Kd0 = Options['Initial']['Kd']
    else:
        epsilon = w[0] / 10
        Kp0 = np.zeros((m, p))
        Ki0 = epsilon * la.pinv(P0[:, :, 0])
        Kd0 = np.zeros((m, p))

    t0 = 0

    maxInterp = Options.get('maxInterp', 10)

    print(f'Optimizing a {m}x{p} PID controller')
    for Iter in range(1, maxInterp + 1):
        print(f'Iteration {Iter}')

        t = np.random.rand(1)
        if 'Structure' in Options:
            Kp = sparsesdpvar(Options['Structure']['Kp'])
            Ki = sparsesdpvar(Options['Structure']['Ki'])
            Kd = sparsesdpvar(Options['Structure']['Kd'])
        else:
            Kp = np.random.rand(m, p)
            Ki = np.random.rand(m, p)
            Kd = np.random.rand(m, p)

        constraints = []
        if 'Sign' in Options:
            constraints.append(Options['Sign']['Kp'] * Kp.flatten() > 0)
            constraints.append(Options['Sign']['Ki'] * Ki.flatten() > 0)
            # constraints.append(Options['Sign']['Kd'] * Kd.flatten() > 0)

        for icase in range(l):  # repeat for each uncertain plant case
            Pcase = P[:, :, icase]
            P0case = P0[:, :, icase]
            for k in range(N):
                wk = w[k]
                s = 1j * wk

                Pk = control.freqresp(Pcase, wk)[0]
                Ck = Kp + (1 / s) * Ki + s / (1 + tau * s) * Kd
                Ck0 = Kp0 + (1 / s) * Ki0 + s / (1 + tau * s) * Kd0

                LMI0 = makeLMI(P0case @ Ki, P0case @ Ki0, t * np.eye(p))

                Z = np.eye(p) + Pk @ Ck
                Z0 = np.eye(p) + Pk @ Ck0
                Y1 = (1 / Smax[k]) * np.eye(p)
                LMI1 = makeLMI(Z, Z0, Y1)

                Y2 = (1 / Tmax[k]) * Pk @ Ck
                LMI2 = makeLMI(Z, Z0, Y2)

                Y3 = (1 / Qmax[k]) * Ck
                LMI3 = makeLMI(Z, Z0, Y3)

                mymagicmargin = 10 ** -15
                margin4 = mymagicmargin * np.eye(4)
                constraints.extend([LMI0 >= margin4, LMI1 >= margin4, LMI2 >= margin4, LMI3 >= margin4])

        def objective(x):
            return -x[0]

        x0 = np.random.rand(1)
        res = minimize(objective, x0, constraints=constraints, method='SLSQP')

        Kp0 = Kp
        Ki0 = Ki
        Kd0 = Kd
        t = res.x[0]

        if abs(t - t0) < 0.001:
            break
        t0 = t

    Kp = Kp0
    Ki = Ki0
    Kd = Kd0

    s = control.TransferFunction.s
    C = Kp + (1 / s) * Ki + s / (1 + tau * s) * Kd
    objval = la.norm(la.inv(P0[:, :, 0] @ Ki), 2)  # spectral norm (largest singular value)

    print(f'Completed after {Iter} iterations! \nObjective value ||(P(0)Ki)^-1||={objval}, t={t}.')
    return C, objval