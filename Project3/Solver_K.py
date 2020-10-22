import os, sys
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

import functions         as func

# -----------------------------------------------------------------------------

class Solver:
    """
    Class for solving a system of two coupled Ordinary Differential Equations (ODEs).
    """
    def __init__(self, f, r0, v0, Np, T, n):
        """
        f: Right-hand side of one of the equations, dv/dv = f(r, dr/dt, t)
        r0: Initial positions,  x   y  [2, nr_planets]
        v0: Initial velocities, vx, vy [2, nr_planets]
        Np: Number of bodies.
        ts: Array with time points.
        """

        #check if f is a callable function
        if not callable(f):
            raise TypeError("ERROR: f is %s, not a function." % type(f))
        self.f = lambda u,t: np.asarray(f(u, t))

        #initial positions and velocities
        self.r0 = r0
        self.v0 = v0

        #number of planets
        self.Np = int(Np)

        #time array
        self.T = T; self.n = n
        self.ts = np.linspace(0, T, n+1)

        #position matrix
        self.r = np.zeros([2, self.ts.size, self.Np])  #[x or y, time step, planet]

        #velocity matrix
        self.v = np.zeros([2, self.ts.size, self.Np])  #[vs or vy, time step, planet]

    def solve(self, method):
        """
        Solves the system of ODEs using either Forward Euler or Velocity Verlet
        methods.
        method: string object with name of desired method.
        """
        #initalize r and v matrices

        print(self.r0)
        print(self.r[:,0,:])

        self.r[:,0,:] = self.r0
        self.v[:,0,:] = self.v0

        #size of time step (use as argument instead of T or n?)
        dt = self.ts[1] - self.ts[0]
        #dt = 1e-3

        #time loop
        for k in range(self.n-1):
            self.k = k  #current index (in time)
            if method == "Euler":
                self.v[:,k+1,:] = self.v[:,k,:] + self.f(self.r[:,k,:], self.ts[k])*dt
                self.r[:,k+1,:] = self.r[:,k,:] + self.v[:,k,:]*dt

            if method == "Verlet":
                self.r[:,k+1,:] = self.r[:,k,:] + self.v[:,k,:]*dt + 0.5*self.f(self.r[:,k,:], self.ts[k])*dt**2
                self.v[:,k+1,:] = self.v[:,k,:] + 0.5*(self.f(self.r[:,k,:], self.ts[k]) + self.f(self.r[:,k+1,:], self.ts[k]))*dt

        return self.r, self.v




if __name__ == '__main__':
    def a(r, t):
        """
        right hand side of dv/dt = -GM/r^2
        Equation derived from Newtonian mechanics
        F = ma
        F = -GMm/r^2
        """
        #M_sun   = 1.989e30              # [kg]
        GM      = 4*np.pi**2            # G*M_sun, Astro units, [AU^3/yr^2]
        unit_r  = r/np.linalg.norm(r)   # Unit vector pointing from sun to Earth
        acc     = -GM/np.linalg.norm(r)**2*unit_r

        return acc

    #init_pos = [1 , 0]     #[AU]
    #init_vel = [0, 2*np.pi]  #[AU/yr]

    T = 10        #[yr]
    n = int(1e4)  #nr of time steps
    n_planet = 1

    #init_pos = np.reshape(init_pos, [2,1])
    #init_vel = np.reshape(init_vel, [2,1])

    #og lese fra inn fil
    if n_planet == 1:
        init_pos = np.array([[1,0]])
        init_vel = np.array([[0,2*np.pi]])

    elif n_planet == 2: # Funker ikke!!!
        init_pos = np.array([[1,0], [2,0]])
        init_vel = np.array([[0,2*np.pi], [0,1]])

    init_pos = np.transpose(init_pos)
    init_vel = np.transpose(init_vel)

    for i in range(n_planet):
        solver1 = Solver(a, init_pos, init_vel, i, T, n)
        solver2 = Solver(a, init_pos, init_vel, i, T, n)
        pos_E, vel_E = solver1.solve(method = "Euler")
        pos_V, vel_V = solver2.solve(method = "Verlet")

        plt.plot(pos_E[0,:-1,0], pos_E[1,:-1,0], label="Euler")
        plt.plot(pos_E[0,0,0], pos_E[1,0,0], "x")
        plt.plot(pos_V[0,:-1,0], pos_V[1,:-1,0], label="Verlet")
        plt.plot(pos_V[0,0,0], pos_V[1,0,0], "x")

    #plt.plot(pos_V[0,0,1], pos_V[1,0,1], "x", color="red",)
    #plt.plot(pos_V[0,:-1,1], pos_V[1,:-1,1], color="red")
    #plt.axis([-2,2, -2,2])
    plt.axis('equal')
    plt.legend()
    plt.show()



    #using the class
    '''
    solver1 = Solver(a, init_pos, init_vel, n_planet, T, n)
    solver2 = Solver(a, init_pos, init_vel, n_planet, T, n)
    pos_E, vel_E = solver1.solve(method = "Euler")
    pos_V, vel_V = solver2.solve(method = "Verlet")
    '''
