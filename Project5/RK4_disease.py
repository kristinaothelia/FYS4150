import numpy as np
import matplotlib.pyplot as plt
import sys

import plots as P

def RK4(a, b, c, x0, y0, fx, fy, N, T, n):
    """
    fx = fS
    fy = fI
    """

    #setting up arrays
    x = np.zeros(n)
    y = np.zeros(n)
    t = np.zeros(n)

    dt = T/n

    #initialize
    x[0] = x0
    y[0] = y0

    #loop for Runge-Kutta 4th Order
    for i in range(n-1):
        kx1 = dt*fx(a, c, t[i], x[i], y[i], N)
        ky1 = dt*fy(a, b, t[i], x[i], y[i], N)

        kx2 = dt*fx(a, c, t[i] + dt/2, x[i] + kx1/2, y[i] + ky1/2, N)
        ky2 = dt*fy(a, b, t[i] + dt/2, x[i] + kx1/2, y[i] + ky1/2, N)

        kx3 = dt*fx(a, c, t[i] + dt/2, x[i] + kx2/2, y[i] + ky2/2, N)
        ky3 = dt*fy(a, b, t[i] + dt/2, x[i] + kx2/2, y[i] + ky2/2, N)

        kx4 = dt*fx(a, c, t[i] + dt, x[i] + kx3, y[i] + ky3, N)
        ky4 = dt*fy(a, b, t[i] + dt, x[i] + kx3, y[i] + ky3, N)

        x[i+1] = x[i] + (kx1 + 2*(kx2 + kx3) + kx4)/6
        y[i+1] = y[i] + (ky1 + 2*(ky2 + ky3) + ky4)/6
        t[i+1] = t[i] + dt

    return x,y,t

def fS(a, c, t, S, I, N):
    """
    Right hand side of S' = dS/dt
    """
    return c*(N - S - I) - a*S*I/N

def fI(a, b, t, S, I, N):
    """
    Right hand side of I' = dI/dt
    """
    return a*S*I/N - b*I

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
