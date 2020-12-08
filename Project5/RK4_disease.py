import numpy as np
import matplotlib.pyplot as plt
import sys

def RK4(x0, y0, fx, fy, n=None, dt=None, T=None):

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
        kx1 = dt*fx(t[i], x[i], y[i]); ky1 = dt*fy(t[i], x[i], y[i])
        kx2 = dt*fx(t[i] + dt/2, x[i] + kx1/2, y[i] + ky1/2); ky2 = dt*fy(t[i] + dt/2, x[i] + kx1/2, y[i] + ky1/2)
        kx3 = dt*fx(t[i] + dt/2, x[i] + kx2/2, y[i] + ky2/2); ky3 = dt*fy(t[i] + dt/2, x[i] + kx2/2, y[i] + ky2/2)
        kx4 = dt*fx(t[i] + dt, x[i] + kx3, y[i] + ky3); ky4 = dt*fy(t[i] + dt, x[i] + kx3, y[i] + ky3)

        x[i+1] = x[i] + (kx1 + 2*(kx2 + kx3) + kx4)/6
        y[i+1] = y[i] + (ky1 + 2*(ky2 + ky3) + ky4)/6
        t[i+1] = t[i] + dt

    return x,y,t

def fS(t, S, I):
    """
    Right hand side of S' = dS/dt
    """
    return c*(N - S - I) - a*S*I/N

def fI(t, S, I):
    """
    Right hand side of I' = dI/dt
    """
    return a*S*I/N - b*I

###data
a = 4    #rate of transmission
c = 0.5  #rate of immunity loss
b = 3
"""
bA = 1   #rate of recovery for population A
bB = 2   #rate of recovery for population B
bC = 3   #rate of recovery for population C
bD = 4   #rate of recovery for population D
"""

#b_list = [bA, bB, bC, bD]  #useful for a loop?

T = 100  #days
n = 1000

N = 400  #nr of individuals in population

S_0 = 300  #initial number of susceptible
I_0 = 100  #initial number of infected

S, I, time = RK4(S_0, I_0, fS, fI, n, T=T)

R = N - S - I

plt.plot(time, S, label="S")
plt.plot(time, I, label="I")
plt.plot(time, R, label="R")
plt.legend()
plt.xlabel("time [days]")
plt.ylabel("nr. of individuals")
plt.show()
