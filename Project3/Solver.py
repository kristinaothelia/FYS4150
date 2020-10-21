import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

import functions            as func

class Solver:
    """
    Class for solving systems of ODEs on the form du/dt = f(u, t).
    """
    def __init__(self, f):
        if not callable(f):
            raise TypeError("ERROR: f is %s, not a function." % type(f))
        self.f = lambda u,t: np.asarray(f(u, t))

    def set_initial_conditions(self, U0):
        #U0 = np.ravel(U0)
        #sys.exit(100)
        #print(U0)
        if isinstance(U0, (float,int)):  #scalar ODE
            self.neq=1
        else:                            #vector ODE
            U0 = np.asarray(U0)
            self.neq = U0.shape
        self.U0 = U0
        #print(U0.shape)

    def solve(self, time_points, method):
        self.t = np.asarray(time_points)
        n = self.t.size
        if self.neq == 1:  #scalar ODE
            self.u = np.zeros(n)
        else:
            self.u = np.zeros((n, self.neq[0], self.neq[1]))

        print(self.neq)
        #sys.exit(100)
        #u = [x, y, vx, vy]
        #assume self.t[0] corresponds to self.U0
        self.u[0,:] = self.U0
        #print(self.u.shape)
        #print(self.U0)

        #time loop
        for k in range(n-1):
            self.k = k
            if method == "Euler":
                self.u[k+1] = self.advance_Euler()
        return self.u, self.t


    def advance_Euler(self):
        """
        euler denne gang
        """
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = t[k+1] - t[k]

        current_vel = u[k, 2:4, :]
        current_pos = u[k, 0:2, :]

        print('ho')
        print(current_pos.shape)
        print(f(current_pos, t).shape)

        new_vel = current_vel + f(current_pos, t)*dt
        new_pos = current_pos + current_vel*dt

        u_new = np.zeros_like(u[k])
        u_new[2:4] = new_vel
        u_new[0:2] = new_pos
        #print(u_new)
        #sys.exit(100)
        #print(u_new.shape)
        return u_new


    # get_acceleration maa kanskje inn i SolarSystem?





if __name__ == '__main__':
    def a(r, t):
        """
        right hand side of dv/dt = -GM/r^2
        Equation derived from Newtonian mechanics
        F = ma
        F = -GMm/r^2
        """
        GM      = 4*np.pi**2            # G*M_sun, Astro units, [AU^3/yr^2]
        #M_sun   = 1.989e30        # [kg]
        unit_r = r/np.linalg.norm(r)  #unit vector pointing from sun to Earth
        #print(unit_r)
        acceleration = -GM/np.linalg.norm(r)**2*unit_r
        #print(acceleration)
        #sys.exit(1)
        return acceleration

    init_pos = [1 , 0]  #[AU]
    init_vel = [0, 2*np.pi]   #[AU/yr] ??

    #using the class
    solver = Solver(a)
    solver.set_initial_conditions([init_pos, init_vel])
    #print(solver.U0)
    T = 10  #[yr]
    dt = 1e-3
    n = int(10e3)
    t = np.linspace(0, T, n+1)
    #print(t)
    u, t = solver.solve(t, method="Euler")
    #print(u)

    plt.plot(u[:,0], u[:,1])
    plt.show()
