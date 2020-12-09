import numpy as np
import matplotlib.pyplot as plt
import sys

def MC(T, N, a, b, c, S0, I0, R0):
    """
    
    """

    dt = np.min([4/(a*N), 1/(b*N), 1/(c*N)])

    time_steps = int(T/dt)
    SIR = np.zeros((time_steps, 3))

    SIR[0, :] = [S0, I0, R0]

    a_0 = a
    w = 4*np.pi/T
    A = 2


    # time loop
    for t in range(time_steps-1):

        SIR[t+1, :] = SIR[t, :]

        N = np.sum(SIR[t, :]) # Update N in case of vital dynamics

        # S -> I
        if np.random.random() < a * SIR[t, 0] * SIR[t, 1] * dt / N:
            SIR[t+1, 0] -= 1
            SIR[t+1, 1] += 1

        # I -> R
        if np.random.random() < b * SIR[t, 1] * dt:
            SIR[t+1, 1] -= 1
            SIR[t+1, 2] += 1

        # R -> S
        if np.random.random() < c * SIR[t, 2] * dt:
            SIR[t+1, 2] -= 1
            SIR[t+1, 0] += 1

    return SIR

a = 4    # rate of transmission
c = 0.5  # rate of immunity loss
b = 3

T = 30  # days
n = 1000

N = 400  # nr of individuals in population

S_0 = 300  # initial number of susceptible
I_0 = 100  # initial number of infected


SIR = MC(T, N, a, b, c, S0=S_0, I0=I_0, R0=0)

print(SIR)

S = SIR[:,0]
I = SIR[:,1]
R = SIR[:,2]

time = np.linspace(0, T, len(S))


plt.plot(time, S, label="S")
plt.plot(time, I, label="I")
plt.plot(time, R, label="R")
plt.legend()
plt.xlabel("time [days]")
plt.ylabel("nr. of individuals")
plt.show()