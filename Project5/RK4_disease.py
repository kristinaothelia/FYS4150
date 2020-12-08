import numpy as np
import matplotlib.pyplot as plt
import sys

def RK4(b, x0, y0, fx, fy, n=None, dt=None, T=None):
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
        kx1 = dt*fx(t[i], x[i], y[i]); ky1 = dt*fy(t[i], x[i], y[i], b)
        kx2 = dt*fx(t[i] + dt/2, x[i] + kx1/2, y[i] + ky1/2); ky2 = dt*fy(t[i] + dt/2, x[i] + kx1/2, y[i] + ky1/2, b)
        kx3 = dt*fx(t[i] + dt/2, x[i] + kx2/2, y[i] + ky2/2); ky3 = dt*fy(t[i] + dt/2, x[i] + kx2/2, y[i] + ky2/2, b)
        kx4 = dt*fx(t[i] + dt, x[i] + kx3, y[i] + ky3); ky4 = dt*fy(t[i] + dt, x[i] + kx3, y[i] + ky3, b)

        x[i+1] = x[i] + (kx1 + 2*(kx2 + kx3) + kx4)/6
        y[i+1] = y[i] + (ky1 + 2*(ky2 + ky3) + ky4)/6
        t[i+1] = t[i] + dt

    return x,y,t

def fS(t, S, I):
    """
    Right hand side of S' = dS/dt
    """
    return c*(N - S - I) - a*S*I/N

def fI(t, S, I, b):
    """
    Right hand side of I' = dI/dt
    """
    return a*S*I/N - b*I

###data
a = 4               # Rate of transmission
c = 0.5             # Rate of immunity loss
b = 3

bA      = 1         # Rate of recovery for population A
bB      = 2         # Rate of recovery for population B
bC      = 3         # Rate of recovery for population C
bD      = 4         # Rate of recovery for population D
b_list  = [bA, bB, bC, bD]

T = 30  #days
n = 1000

N = 400  #nr of individuals in population

S_0 = 300  #initial number of susceptible
I_0 = 100  #initial number of infected

S, I, time = RK4(b, S_0, I_0, fS, fI, n, T=T)

R = N - S - I

def plot_SIR(time, b, S, I, R, T, method, save_plot=False):

    plt.figure()
    plt.plot(time, S, label="Susceptible")
    plt.plot(time, I, label="Infected")
    plt.plot(time, R, label="Recovered")
    plt.legend(fontsize=15)
    plt.title('??? b=%g' %b)
    plt.xlabel("Time [days]", fontsize=15)
    plt.ylabel("Nr. of individuals", fontsize=15)
    plt.xticks(fontsize=13);plt.yticks(fontsize=13)
    plt.tight_layout()

    if save_plot:
        print('\nSaving plot for method: %s, T=%g, b=%g' %(method, T, b))
        plt.savefig('Results/SIR_%s_T[%g]_b[%g]'% (method, T, b))
    else:
        plt.show()


for i in range(len(b_list)):
    S, I, time  = RK4(b_list[i], S_0, I_0, fS, fI, n, T=T)
    R           = N - S - I
    plot_SIR(time, b_list[i], S, I, R, T, method='RK4', save_plot=True)
