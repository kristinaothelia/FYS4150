import numpy as np
import matplotlib.pyplot as plt
import sys

import plots as P

e = 0.25  #birth
d = 0.2  #death
dI = 0.35   #death from disease

f = 0.5  #vaccination rate

def RK4(a_in, b, c, x0, y0, z0, N, T, n, fx, fy, fz=None, Vital=False, seasonal=False, vaccine=False):
    """
    4th Order Runge-Kutta method for solving a system of three coupled
    differential equations.

    Vital: True/False
    Include death rates, birth rates and death rates due to disease.

    fx = fS
    fy = fI
    """

    #setting up arrays
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    t = np.zeros(n)

    #size of time step
    dt = T/n

    #initialize
    x[0] = x0
    y[0] = y0
    z[0] = z0

    #loop for Runge-Kutta 4th Order
    if Vital:
    #oppgave c
        for i in range(n-1):

            if seasonal:
                #oppgave d
                a0 = a_in
                A = 4
                #omega = 4*np.pi/T  #oscillate once per year????
                omega = 0.5  #how to interpret?
                a = A*np.cos(omega*t[i]) + a0
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

# e  - birth rate
# d  - death rate
# dI - death rate of infected people due to the disease

def fS(a, b, c, N, S, I, R=None, vital=False, vaccine=False): # Kan ogsaa adde if test for seasonal osv...
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



'''
###data
a = 4               # Rate of transmission
c = 0.5             # Rate of immunity loss
b = 3

T = 30  #days
n = 1000

N = 400  #nr of individuals in population

S_0 = 300  #initial number of susceptible
I_0 = 100  #initial number of infected

S, I, time = RK4(b, S_0, I_0, fS, fI, n, T=T)

R = N - S - I

P.plot_SIR(time, b, S, I, R, T, method='RK4')
'''
