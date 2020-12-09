import numpy as np
import matplotlib.pyplot as plt
import sys


def MC(a, b, c, S_0, I_0, R_0, N, T):
    """

    """

    #size of time step
    dt = np.min([4/(a*N), 1/(b*N), 1/(c*N)])

    #nr of time steps
    N_time = int(T/dt)

    #set up empty arrys
    S = np.zeros(N_time)
    I = np.zeros_like(S)
    R = np.zeros_like(S)

    #initalize arrays
    S[0] = S_0
    I[0] = I_0
    R[0] = R_0


    """
    a_0 = a
    w = 4*np.pi/T
    A = 2
    """


    # time loop
    for i in range(N_time - 1):

        S[i+1] = S[i]
        I[i+1] = I[i]
        R[i+1] = R[i]

        #N = np.sum(SIR[t, :]) # Update N in case of vital dynamics

        # S -> I
        r_SI = np.random.random()
        if r_SI < (a*S[i]*I[i]*dt/N):
            S[i+1] -= 1
            I[i+1] += 1

        # I -> R
        r_IR = np.random.random()
        if r_IR < (b*I[i]*dt):
            I[i+1] -= 1
            R[i+1] += 1

        # R -> S
        r_RS = np.random.random()
        if r_RS < (c*R[i]*dt):
            R[i+1] -= 1
            S[i+1] += 1

    return S, I, R

A = False
if A:
    a = 4    # rate of transmission
    c = 0.5  # rate of immunity loss
    b = 1

    T = 30  # days
    n = 1e6

    N = 400  # nr of individuals in population

    S_0 = 300  # initial number of susceptible
    I_0 = 100  # initial number of infected


    S, I, R = MC(T, N, a, b, c, S_0=S_0, I_0=I_0, R_0=0)

    time = np.linspace(0, T, len(S))

    plt.plot(time, S, label="S")
    plt.plot(time, I, label="I")
    plt.plot(time, R, label="R")
    plt.legend()
    plt.xlabel("time [days]")
    plt.ylabel("nr. of individuals")
    plt.show()
