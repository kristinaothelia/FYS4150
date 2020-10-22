import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

import functions            as func

class Solver:
    """
    Class for solving a system of two coupled Ordinary Differential Equations (ODEs).
    """
    def __init__(self, f, M, r0, v0, Np, T, n):
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

        self.M  = M

        self.M_Sun = 1.989*10**30  # [kg]

        #initial positions and velocities
        self.r0 = r0
        self.v0 = v0

        #number of planets
        self.Np = int(Np)

        #time array
        self.T = T; self.n = n
        self.ts = np.linspace(0, T, n+1)

        #position matrix
        self.r = np.zeros([2, self.ts.size-1, self.Np])  #[x or y, time step, planet]

        #velocity matrix
        self.v = np.zeros([2, self.ts.size-1, self.Np])  #[vs or vy, time step, planet]


    def solve(self, method):
        """
        Solves the system of ODEs using either Forward Euler or Velocity Verlet
        methods.
        method: string object with name of desired method.
        """
        #initalize r and v matrices
        self.r[:,0,:] = self.r0
        self.v[:,0,:] = self.v0

        #size of time step (use as argument instead of T or n?)
        dt = self.ts[1] - self.ts[0]

        '''
        #time loop
        for k in range(self.n-1):
            self.k = k  #current index (in time)
            if method == "Euler":
                self.v[:,k+1,:] = self.v[:,k,:] + self.f(self.r[:,k,:], self.ts[k])*dt
                self.r[:,k+1,:] = self.r[:,k,:] + self.v[:,k,:]*dt
            if method == "Verlet":
                self.r[:,k+1,:] = self.r[:,k,:] + self.v[:,k,:]*dt + 0.5*self.f(self.r[:,k,:], self.ts[k])*dt**2
                self.v[:,k+1,:] = self.v[:,k,:] + 0.5*(self.f(self.r[:,k,:], self.ts[k]) + self.f(self.r[:,k+1,:], self.ts[k+1]))*dt
        '''

        for k in range(self.n-1):
            self.k = k  #current index (in time)
            acceleration1 = np.zeros((2, self.Np))

            for n in range(self.Np):
                #self.n = n
                for i in range(self.Np):
                    if i  != n:
                        temp_r = self.r[:,k,n] - self.r[:,k,i]
                        unit_r = temp_r/np.linalg.norm(temp_r, axis=0)
                        acceleration1[:,n] = (4*np.pi**2*self.M[i]/self.M_Sun)/np.linalg.norm(temp_r, axis=0)**2
                    else:
                        pass


            if method == "Euler":
                    self.v[:,k+1,:] = self.v[:,k,:] + acceleration*dt
                    self.r[:,k+1,:] = self.r[:,k,:] + self.v[:,k,:]*dt

            if method == "Verlet":
                    self.r[:,k+1,:] = self.r[:,k,:] + self.v[:,k,:]*dt + 0.5*acceleration1*dt**2
                    
                    acceleration2 = np.zeros((2, self.Np))
                    for n in range(self.Np):
                        #self.n = n
                        for i in range(self.Np):
                            if i  != n:
                                temp_r = self.r[:,k+1,n] - self.r[:,k+1,i]
                                unit_r = temp_r/np.linalg.norm(temp_r, axis=0)
                                acceleration2[:,n] = (4*np.pi**2*self.M[i]/self.M_Sun)/np.linalg.norm(temp_r, axis=0)**2
                            else:
                                pass
                    
                    self.v[:,k+1,:] = self.v[:,k,:] + 0.5*(acceleration1+acceleration2)*dt

        return self.r, self.v, self.ts


if __name__ == '__main__':
    print("Import into main.py.")
