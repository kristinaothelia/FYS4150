import numpy as np
import matplotlib.pyplot as plt
import sys

e = 0.25  #birth
d = 0.2  #death
dI = 0.35   #death from disease

def MC(a_in, b, c, S_0, I_0, R_0, N, T, vitality=False, seasonal=False):
    """

    """

    if seasonal:
        #oppgave d
        a0 = a_in
        A = 4
        #omega = 4*np.pi/T  #oscillate once per year????
        omega = 0.5  #how to interpret?
        a = A*np.cos(omega*0) + a0

        #size of time step
        dt = np.min([4/(a*N), 1/(b*N), 1/(c*N)])

        #nr of time steps
        N_time = int(T/dt)

        #set up empty arrys
        S = np.zeros(N_time)
        I = np.zeros_like(S)
        R = np.zeros_like(S)

    else:
        a = a_in

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

        if seasonal:
            #oppgave d
            a0 = a_in
            A = 4
            #omega = 4*np.pi/T  #oscillate once per year????
            omega = 0.5  #how to interpret?
            a = A*np.cos(omega*i) + a0
        else:
            a = a_in

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

        if vitality:
            #death rate d in general population S, I and R
            r_dS = np.random.random()
            if r_dS < (d*S[i]*dt):     #d*S*dt = probability of one individual dying in S category
                S[i+1] -= 1

            r_dI = np.random.random()
            if r_dS < (d*I[i]*dt):
                I[i+1] -= 1

            r_dR = np.random.random()
            if r_dR < (d*R[i]*dt):
                R[i+1] -= 1

            #death rate dI for infected population I
            r_dII = np.random.random()
            if r_dII < (dI*I[i]*dt):
                I[i+1] -= 1

            #birth rate e for general population S, I and R
            r_eS = np.random.random()
            if r_eS < (e*S[i]*dt):     #e*S*dt = probability of one individual being born in S category
                S[i+1] += 1

            r_eI = np.random.random()
            if r_eS < (e*I[i]*dt):
                I[i+1] += 1

            r_eR = np.random.random()
            if r_eR < (e*R[i]*dt):
                R[i+1] += 1

    return S, I, R

"""
#kanskje fjerne hele denne?
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
"""
