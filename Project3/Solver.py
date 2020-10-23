"""
FYS4150 - Project 3: Solver clss
"""
from __future__          import division
import os, sys
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

class Solver:
    """
    Class for solving a system of two coupled Ordinary Differential Equations (ODEs).
    """
    def __init__(self, M, r0, v0, Np, T, n):
        """
        M : Planet masses
        r0: Initial positions,  x   y  [2, nr_planets]
        v0: Initial velocities, vx, vy [2, nr_planets]
        Np: Number of bodies.
        ts: Array with time points.
        """

        self.M     = M
        self.M_Sun = 1.989*10**30  # [kg]
        self.GM    = 4*np.pi**2            # G*M_sun, Astro units, [AU^3/yr^2]
        self.G     = self.GM/self.M_Sun

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


    def acc_sun_in_motion(self, k_val, beta):

        acceleration = np.zeros((2, self.Np))

        for n in range(self.Np):

            acceleration_sum = 0
            for i in range(self.Np):
                if i != n:
                    temp_r = self.r[:,k_val,n] - self.r[:,k_val,i]
                    unit_r = temp_r/np.linalg.norm(temp_r, axis=0)
                    acceleration_sum -= (self.G*self.M[i])/np.linalg.norm(temp_r, axis=0)**beta*unit_r
                else:
                    pass

            acceleration[:,n] = acceleration_sum
        return acceleration



    def acceleration_func(self, k_val, beta):
        acceleration = np.zeros((2, self.Np))

        for n in range(self.Np):
            #self.n = n
            acceleration_sum = 0
            for i in range(self.Np):
                if i != n:
                    temp_r = self.r[:,k_val,n] - self.r[:,k_val,i]
                    unit_r = temp_r/np.linalg.norm(temp_r, axis=0)
                    #acceleration_sum += (self.G*self.M[i])/np.linalg.norm(temp_r, axis=0)**2*unit_r
                    acceleration_sum += (self.G*self.M[i])/np.linalg.norm(temp_r, axis=0)**beta*unit_r
                else:
                    pass

            unit_r_sun = self.r[:,k_val,n]/np.linalg.norm(self.r[:,k_val,n], axis=0)
            #acceleration[:,n] = acceleration_sum - self.G*self.M_Sun/np.linalg.norm(self.r[:,k_val,n], axis=0)**2*unit_r_sun
            acceleration[:,n] = acceleration_sum - self.G*self.M_Sun/np.linalg.norm(self.r[:,k_val,n], axis=0)**beta*unit_r_sun
        return acceleration


    def solve(self, method, beta=2, SunInMotion=False):
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

        if SunInMotion == True:

            for k in range(self.n-1):
                self.k = k  #current index (in time)

                acceleration1 = self.acc_sun_in_motion(k, beta)


                if method == "Euler":
                        self.v[:,k+1,:] = self.v[:,k,:] + acceleration1*dt
                        self.r[:,k+1,:] = self.r[:,k,:] + self.v[:,k,:]*dt

                if method == "Verlet":
                        self.r[:,k+1,:] = self.r[:,k,:] + self.v[:,k,:]*dt + 0.5*acceleration1*dt**2
                        acceleration2 = self.acc_sun_in_motion(k+1, beta)
                        self.v[:,k+1,:] = self.v[:,k,:] + 0.5*(acceleration1+acceleration2)*dt

        else:

            for k in range(self.n-1):
                self.k = k  #current index (in time)

                acceleration1 = self.acceleration_func(k, beta)

                if method == "Euler":
                        self.v[:,k+1,:] = self.v[:,k,:] + acceleration1*dt
                        self.r[:,k+1,:] = self.r[:,k,:] + self.v[:,k,:]*dt

                if method == "Verlet":
                        self.r[:,k+1,:] = self.r[:,k,:] + self.v[:,k,:]*dt + 0.5*acceleration1*dt**2
                        acceleration2 = self.acceleration_func(k+1, beta)
                        self.v[:,k+1,:] = self.v[:,k,:] + 0.5*(acceleration1+acceleration2)*dt

        return self.r, self.v, self.ts


if __name__ == '__main__':
    print("Import into main.py.")
