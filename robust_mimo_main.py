import numpy as np
import matplotlib.pyplot as plt
from control import tf, dcgain, frd, pade, bode
from auto_mimo_pid import auto_mimo_pid
from mimo_pid_convert_auto import auto_mimo_pid_convert
#, mintegraltf

# Define the transfer function matrix P

s = tf('s')

# P = np.array([[12.8*tf([1], [1, 16.7], delay=1)/(16.7*s+1), -18.9*tf([1], [1, 21], delay=3)/(21*s+1)],
#               [6.6*tf([1], [1, 10.9], delay=7)/(10.9*s+1), -19.4*tf([1], [1, 14.2], delay=3)/(14.2*s+1)]])
P = tf([[[12.8], [-18.9]], [[6.6], [-19.4]]], 
               [[[16.7, 1], [21, 1]], [[10.9 ,1], [14.2, 1]]])
# delay_p = tf([[[1], [0]], [[7], [3]]], 
#         [[[1], [1]], [[1], [1]]])
[num1, den1] = pade(1, 1)
[num2, den2] = pade(3, 1)
[num3, den3] = pade(7, 1)
[num4, den4] = pade(3, 1)
pade_estimation = tf([[num1, num2], [num3, num4]], 
        [[den1, den2], [den3, den4]])
print(P)
print(pade_estimation)
P = P * pade_estimation
# print(P)  
# bodeplot of the transfer function matrix P


#
N = 300
w = np.logspace(-3, 3, N)
P0 = dcgain(P)
tau = 0.3

# Specifications
Smax = 1.4 * np.ones(N)
Tmax = 1.4 * np.ones(N)
Qmax = 3 / np.min(np.linalg.svd(dcgain(P))[1]) * np.ones(N)

# Options (not used in this example)
# Option = {'Structure': {'Kp': np.eye(2), 'Ki': np.eye(2), 'Kd': np.eye(2)},
#           'Initial': {'Kp': np.array([[0.001, 0], [0, -0.001]]),
#                       'Ki': np.array([[0.001, 0], [0, -0.001]]),
#                       'Kd': np.zeros((2, 2))}}
Option = {}

# Auto-tune the MIMO PID controller
# G = mintegraltf(P, w, Smax, Tmax, Qmax, tau, Option)
G = auto_mimo_pid(P, w, Smax, Tmax, Qmax, tau, Option)
#G = auto_mimo_pid_convert(P, w, Smax, Tmax, Qmax, tau, Option)
# Analysis
L = frd(P * G, w)
S = (np.eye(2) + P * G).inverse()
T = P * G * S
Q = G * S

# Plotting
plt.figure(1)
plt.loglog(w, np.maximum.reduce(np.abs(S)), label='S')
plt.title('Sensitivity Function')
plt.loglog(w, 20*np.log10(Smax), 'r', label='Smax')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')
plt.legend()

plt.figure(2)
plt.loglog(w, np.maximum.reduce(np.abs(T)), label='T')
plt.title('Complementary Sensitivity Function')
plt.loglog(w, 20*np.log10(Tmax), 'r', label='Tmax')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')

plt.legend()

plt.figure(3)
plt.loglog(w, np.maximum.reduce(np.abs(Q)), label='Q')
plt.loglog(w, 20*np.log10(Qmax), 'r', label='Qmax')
plt.title('Control Effort')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')
plt.legend()

plt.show()