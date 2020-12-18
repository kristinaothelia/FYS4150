import sys
import numpy             as np
import matplotlib.pyplot as plt
import plots             as P

e  = 0.25       # Birth rate
d  = 0.2        # Death rate
dI = 0.35       # Death rate of infected people due to the disease
f  = 0.5        # Vaccination rate


# Runge-Kutta 4
# ----------------------------------------------------------------------------

def RK4(a_in, b, c, x0, y0, z0, N, T, n, fx, fy, fz=None, Vital=False, seasonal=False, vaccine=False):
    """
    4th Order Runge-Kutta method for solving a system of three coupled
    differential equations.

    Vital: True/False
    Include death rates, birth rates and death rates due to disease.

    fx = fS
    fy = fI
    """

    # Setting up arrays
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    t = np.zeros(n)

    # Size of time step
    dt = T/n

    # Initialize
    x[0] = x0
    y[0] = y0
    z[0] = z0

    # Loop for Runge-Kutta 4th Order
    if Vital:     # ex. c)
        for i in range(n-1):

            if seasonal:    # ex. d)
                a0    = a_in
                A     = 4
                #omega = 4*np.pi/T  #oscillate once per year????
                omega = 0.5  #how to interpret?
                a     = A*np.cos(omega*t[i]) + a0
            else:
                a = a_in

            kx1 = dt*fx(a, b, c, N, x[i], y[i], z[i], vital=True)
            ky1 = dt*fy(a, b, c, N, x[i], y[i], z[i], vital=True)
            kz1 = dt*fz(a, b, c, N, x[i], y[i], z[i], vital=True)

            kx2 = dt*fx(a, b, c, N, x[i] + kx1/2, y[i] + ky1/2, z[i] + ky1/2, vital=True)
            ky2 = dt*fy(a, b, c, N, x[i] + kx1/2, y[i] + ky1/2, z[i] + kz1/2, vital=True)
            kz2 = dt*fz(a, b, c, N, x[i] + kx1/2, y[i] + ky1/2, z[i] + kz1/2, vital=True)

            kx3 = dt*fx(a, b, c, N, x[i] + kx2/2, y[i] + ky2/2, z[i] + kz2/2, vital=True)
            ky3 = dt*fy(a, b, c, N, x[i] + kx2/2, y[i] + ky2/2, z[i] + kz2/2, vital=True)
            kz3 = dt*fz(a, b, c, N, x[i] + kx2/2, y[i] + ky2/2, z[i] + kz2/2, vital=True)

            kx4 = dt*fx(a, b, c, N, x[i] + kx3, y[i] + ky3, z[i] + kz3, vital=True)
            ky4 = dt*fy(a, b, c, N, x[i] + kx3, y[i] + ky3, z[i] + kz3, vital=True)
            kz4 = dt*fz(a, b, c, N, x[i] + kx3, y[i] + ky3, z[i] + kz3, vital=True)

            x[i+1] = x[i] + (kx1 + 2*(kx2 + kx3) + kx4)/6
            y[i+1] = y[i] + (ky1 + 2*(ky2 + ky3) + ky4)/6
            z[i+1] = z[i] + (kz1 + 2*(kz2 + kz3) + kz4)/6
            t[i+1] = t[i] + dt

    else:
        #oppgave a og b
        for i in range(n-1):

            if seasonal:
                #oppgave d
                a0 = a_in
                A = 3
                #omega = 4*np.pi/T  #oscillate once per year????
                omega = 0.5
                a = A*np.cos(omega*t[i]) + a0
            else:
                a = a_in

            if vaccine:

                kx1 = dt*fx(a, b, c, N, x[i], y[i], vaccine=True)
                ky1 = dt*fy(a, b, c, N, x[i], y[i], vaccine=True)

                kx2 = dt*fx(a, b, c, N, x[i] + kx1/2, y[i] + ky1/2, vaccine=True)
                ky2 = dt*fy(a, b, c, N, x[i] + kx1/2, y[i] + ky1/2, vaccine=True)

                kx3 = dt*fx(a, b, c, N, x[i] + kx2/2, y[i] + ky2/2, vaccine=True)
                ky3 = dt*fy(a, b, c, N, x[i] + kx2/2, y[i] + ky2/2, vaccine=True)

                kx4 = dt*fx(a, b, c, N, x[i] + kx3, y[i] + ky3, vaccine=True)
                ky4 = dt*fy(a, b, c, N, x[i] + kx3, y[i] + ky3, vaccine=True)

                x[i+1] = x[i] + (kx1 + 2*(kx2 + kx3) + kx4)/6
                y[i+1] = y[i] + (ky1 + 2*(ky2 + ky3) + ky4)/6
                t[i+1] = t[i] + dt

            else:

                kx1 = dt*fx(a, b, c, N, x[i], y[i])
                ky1 = dt*fy(a, b, c, N, x[i], y[i])

                kx2 = dt*fx(a, b, c, N, x[i] + kx1/2, y[i] + ky1/2)
                ky2 = dt*fy(a, b, c, N, x[i] + kx1/2, y[i] + ky1/2)

                kx3 = dt*fx(a, b, c, N, x[i] + kx2/2, y[i] + ky2/2)
                ky3 = dt*fy(a, b, c, N, x[i] + kx2/2, y[i] + ky2/2)

                kx4 = dt*fx(a, b, c, N, x[i] + kx3, y[i] + ky3)
                ky4 = dt*fy(a, b, c, N, x[i] + kx3, y[i] + ky3)

                x[i+1] = x[i] + (kx1 + 2*(kx2 + kx3) + kx4)/6
                y[i+1] = y[i] + (ky1 + 2*(ky2 + ky3) + ky4)/6
                t[i+1] = t[i] + dt

    return x,y,z,t


def fS(a, b, c, N, S, I, R=None, vital=False, vaccine=False): 
    """
    Right hand side of S' = dS/dt
    """
    if vital:
        temp = c*R - a*S*I/N - d*S + e*N
    elif vaccine:
        R = N - S - I
        temp = c*R - a*S*I/N - f
    else:
        temp = c*(N-S-I) - a*S*I/N

    return temp

def fI(a, b, c, N, S, I, R=None, vital=False, vaccine=False):
    """
    Right hand side of I' = dI/dt
    """
    if vital:
        temp = a*S*I/N - b*I - d*I - dI*I
    elif vaccine:
        temp = a*S*I/N - b*I
    else:
        temp = a*S*I/N - b*I

    return temp

def fR(a, b, c, N, S, I, R, vital=False, vaccine=False): # Ikke endret ennaa
    """
    Right hand side of I' = dI/dt
    """
    if vital:
        temp = b*I - c*R - d*R
    elif vaccine:
        R = N - S - I
        temp = b*I - c*R + f
    else:
        temp = 0

    return temp



# Monte Carlo
# ----------------------------------------------------------------------------

def MC(a_in, b, c, S_0, I_0, R_0, N, T, vitality=False, seasonal=False):
    """

    """

    if seasonal:    # ex. d)

        a0     = a_in
        A      = 4
        #omega  = 4*np.pi/T  #oscillate once per year????
        omega  = 0.5  #how to interpret?
        a      = A*np.cos(omega*0) + a0

        # Size of time step
        dt     = np.min([4/(a*N), 1/(b*N), 1/(c*N)])

        # Nr of time steps
        N_time = int(T/dt)

        # Set up empty arrys
        S = np.zeros(N_time)
        I = np.zeros_like(S)
        R = np.zeros_like(S)

    else:
        a  = a_in

        # Size of time step
        dt = np.min([4/(a*N), 1/(b*N), 1/(c*N)])

        # Nr of time steps
        N_time = int(T/dt)

        # Set up empty arrys
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

        if seasonal:    # ex. d)
            a0 = a_in
            A  = 4
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