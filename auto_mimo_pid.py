import numpy as np
import matplotlib.pyplot as plt
import control as ct
from control import tf, dcgain, frd, pade, bode, freqresp
from scipy import signal
import time
import cvxpy as cp


def makeLMI(Z, Z0, Y, p):
    """Constructs the LMI"""
    # note this function I encountered a bug when first implement it, where I use np.array instead of bmat
    # also the constraints should be real, so I added the cp.real() function
    # indentity matrix of size p
    P = np.eye(p)
    
    LMI = cp.bmat([[Z.T @ Z0 + Z0.T @ Z - Z0.T @ Z0, Y.T],
                     [Y, P]])
    LMI = cp.real(LMI)
    #don't allow complex numbers
    return LMI


def auto_mimo_pid(P,w,Smax,Tmax,Qmax,tau,Options ):

                  
    """
    Autotune a MIMO PID for a nominal plant by LMIs.

    Args:
        P: Plant in ss, tf, zpk, or frd form; input an array of plants to account for uncertainty and variations in parameters
        w: Vector of N frequencies responsibly chosen to cover the dynamic range
        Smax: Sensitivity specification upper bound
        Tmax: Low frequency sensitivity specification upper bound
        Qmax: Cost of feedback specification upper bound
        tau: Derivative action time constant
        Options: Optional dict which specifies additional options

    Returns:
        C: Optimized MIMO PID controller transfer matrix
        objval: Value of the objective after optimization
    """
    p, m = P(0).shape  # Number of inputs and outputs

    l = p*m  # Number of plants


    N = len(w)
    # PW0 = np.zeros((p,m,l), dtype=complex)
    Pw0 =  freqresp(P, w[0]).fresp# frequency response at w[0], take  you can use the response_data attribute to extract the frequency response data.
    P0 = Pw0
  
    # if len(sys.argv) < 7:
    #     Options = {}
    if Options is None:
        Options = {}
    # initialize the PID controller
    if 'Initial' in Options:
        Kp0 = Options['Initial']['Kp']
        Ki = Options['Initial']['Ki']
        Kd = Options['Initial']['Kd']

    else:
        eplison = 0.01
        #eplison = w[0] / 10
        Kp0 = np.zeros((m,p))
        Ki0 = eplison * np.linalg.pinv(P0[:,:,0],rcond=1e-15)
        Ki0 = np.real(Ki0.reshape(2, 2))

        Kd0 = np.zeros((m,p))
    t0 = 0

    if 'maxInterp' in Options:  # specify max number of interpolations
        maxInterp = Options['maxInterp']
    else:
        maxInterp = 2
    
    print(f'Optimizaing a {m}x{p} PID controller')
    start_time = time.time()

    #start iteration
    for Iter in range(1,maxInterp +1):
        print(f'Iteration {Iter} of {maxInterp}')
        t = cp.Variable()
        if 'Structure' in Options:
            Kp = cp.Variable(Options['Structure']['Kp'].shape, PSD=True)
            Ki = cp.Variable(Options['Structure']['Ki'].shape, PSD=True)
            Kd = cp.Variable(Options['Structure']['Kd'].shape, PSD=True)
        else:
            Kp = cp.Variable((m,p), symmetric=True)
            Ki = cp.Variable((m,p), symmetric=True)
            Kd = cp.Variable((m,p), symmetric=True)

        if 'Sign' in Options:
            constraints = [
                cp.multiply(Options['Sign']['Kp'], Kp) > 0,
                cp.multiply(Options['Sign']['Ki'], Ki) > 0,
                cp.multiply(Options['Sign']['Kd'], Kd) > 0
            ]

        else:
            constraints = [
                # Kp > 0,
                # Ki > 0,
                # Kd > 0
            ]
            

         #loop over all plants 
        # Plist = P.returnScipySignalLTI() 
        # icase = 0
        # for outer_list in Plist:
        #     for tf in outer_list:
        #         num = tf.num
        #         den = tf.den
        #         PCase = control.TransferFunction(num, den)
                
        #         P0Case = P0[:,:,icase]
        #         icase = icase + 1 if icase <= l else 0
        PCase = P
        P0Case = P0
        P0Case = P0Case.reshape(2, 2)
        for K in range(N):
            wk = w[K]
            s  = 1j * wk
            Pk = freqresp(PCase,wk).fresp
            
            Ck = Kp + (1/s) * Ki + s/(1+ tau*s) * Kd
            CK0 = Kp0 + (1/s) * Ki0 + s/(1+ tau*s) * Kd0
            # LMI0 = np.zeros((p + p, p + p), dtype=complex)
            LMI0 = makeLMI(P0Case @ Ki, P0Case @ Ki0,  t * np.eye(p),p)
            Pk = Pk.reshape(2, 2)
            # Ck = Ck.reshape(2, 2)
            # CK0 = CK0.reshape(2, 2)
            Z = np.eye(p) + Pk @ Ck
            Z0 = np.eye(p) + Pk @ CK0
            Y1 = (1 /Smax[K]) * np.eye(p)
            LMI1 = makeLMI(Z, Z0, Y1,p)
            Y2 = (1 / Tmax[K]) * Pk @ Ck
            LMI2 = makeLMI(Z, Z0, Y2,p)
            Y3 = (1 / Qmax[K]) * Ck
            LMI3 = makeLMI(Z, Z0, Y3,p)
            mymagic  = 1e-15
            margin4 = mymagic * np.eye(4)
            #make LMI non complex
            # LMI0 = np.real(LMI0)
            # LMI1 = np.real(LMI1)
            # LMI2 = np.real(LMI2)
            # LMI3 = np.real(LMI3)

            constraints.extend([LMI0 >= margin4, LMI1 >= margin4, LMI2 >= margin4, LMI3 >= margin4])
      
            print(f'K={K+1},number of constraints={len(constraints)}, current evaluating frequency={wk}')    
            objective = cp.Minimize(-t)
            problem = cp.Problem(objective, constraints)
            # result = problem.solve(solver=cp.SCS)
            result = problem.solve(solver=cp.OSQP)
            # diagnostics = {
            #     'status': problem.status,
            #     'optimal_value': problem.value,
            #     'optimal_var': t.value,}
            # Kp0 = float(Kp)
            # Ki0 = float(Ki)
            # Kd0 = float(Kd)
            #t = float(t)
        # eplison = 1e-3
        # tt = cp.abs(t.value)
        # print(f't={tt}')
        # if tt < eplison:
        #     break

    t0 = t

    Kp = Kp0
    Ki = Ki0
    Kd = Kd0
    print(f'Kp={Kp}, Ki={Ki}, Kd={Kd}')

    # s = tf('s')
    # # construct the PID controller for mimo system
    # C = Kp + (1 / s) * Ki + s / (1 + tau * s) * Kd

    # initialize Ckp, Cki, Ckd
    # Ckp= np.zeros((m,p))
    # Cki = np.zeros((m,p))
    # Ckd = np.zeros((m,p))
    # initialize Ckp, Cki, Ckd
    Ckp = np.zeros((m, p))
    Cki = np.zeros((m, p))
    Ckd = np.zeros((m, p))

    # restoring the PID controller
    for i in range(m):
        for j in range(p):
            Ckp[i, j] = Kp[i, j]
            Cki[i, j] = Ki[i, j]
            Ckd[i, j] = Kd[i, j]

    C11_num = [Ckp[0, 0], Cki[0, 0], Ckd[0, 0]]
    C11_den = [1, tau]
    C12_num = [Ckp[0, 1], Cki[0, 1], Ckd[0, 1]]
    C12_den = [1, tau]
    C21_num = [Ckp[1, 0], Cki[1, 0], Ckd[1, 0]]
    C21_den = [1, tau]
    C22_num = [Ckp[1, 1], Cki[1, 1], Ckd[1, 1]]
    C22_den = [1, tau]

    C = tf([[C11_num, C12_num], [C21_num, C22_num]],
        [[C11_den, C12_den], [C21_den, C22_den]])  
    print(f'C={C}')
    objval = np.linalg.norm(np.linalg.inv(P0[:, :, 0] @ Ki),ord=2)  # spectra norm (largest singular value)

    # iteration = Iter - 1
    elapsed_time = time.time() - start_time
    print(f"Completed after {Iter} iterations!")
    print(f"Objective value ||(P(0)Ki)^-1||={objval}, t={t}.")
    print(f"Total time: {elapsed_time} seconds")
                
    return C, objval