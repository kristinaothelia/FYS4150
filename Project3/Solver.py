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
        M : Planet masses.
        r0: Initial positions,  x   y  [2, nr_planets].
        v0: Initial velocities, vx, vy [2, nr_planets].
        Np: Number of bodies.
        ts: Array with time points.
        """

        self.M     = M
        self.M_Sun = 1.989*10**30        # [kg]
        self.GM    = 4*np.pi**2          # [AU^3/yr^2] (G*M_sun, Astro units)
        self.G     = self.GM/self.M_Sun
        #print(self.G)
        self.c     = 63239.7             # [AU/yr]

        # initial positions and velocities
        self.r0 = r0
        self.v0 = v0
        #print(self.r0)

        # number of planets
        self.Np = int(Np)

        # time array
        self.T  = T
        self.n  = n # int(n)
        self.ts = np.linspace(0, T, n+1)

        # position matrix
        self.r = np.zeros([2, self.ts.size-1, self.Np])  #[x or y, time step, planet]

        # velocity matrix
        self.v = np.zeros([2, self.ts.size-1, self.Np])  #[vs or vy, time step, planet]


    def acc_sun_in_motion(self, k_val, beta):
        """ Calculates acceleration according to mass center """

        acceleration = np.zeros((2, self.Np))

        for n in range(self.Np):

            acceleration_sum = 0
            for i in range(self.Np):
                #print(i)
                if i != n:
                    temp_r = self.r[:,k_val,n] - self.r[:,k_val,i]
                    unit_r = temp_r/np.linalg.norm(temp_r, axis=0)
                    acceleration_sum -= (self.G*self.M[i])/np.linalg.norm(temp_r, axis=0)**beta*unit_r
                else:
                    pass

            acceleration[:,n] = acceleration_sum
        return acceleration



    def acceleration_func(self, k_val, beta):
        """ Calculates acceleration when assuming fixed Sun"""
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
            acceleration[:,n] = acceleration_sum - self.G*self.M_Sun/np.linalg.norm(self.r[:,k_val,n], axis=0)**beta*unit_r_sun
        return acceleration


    def relativity(self, k_val, beta=2):
        """Calculates acceleration with relativistic correction (fixed sun)"""


        acceleration = np.zeros((2, self.Np))

        for n in range(self.Np):

            acceleration_sum = 0
            for i in range(self.Np):
                if i != n:
                    temp_r = self.r[:,k_val,n] - self.r[:,k_val,i]
                    unit_r = temp_r/np.linalg.norm(temp_r, axis=0)
                    acceleration_sum += (self.G*self.M[i])/np.linalg.norm(temp_r, axis=0)**beta*corr*unit_r
                else:
                    pass

            l = np.cross(self.r[:,k_val,n], self.v[:,k_val,n], axis=0) # angular momentum
            l = np.linalg.norm(l)
            r_sun = self.r[:,k_val,n]
            unit_r_sun = self.r[:,k_val,n]/np.linalg.norm(self.r[:,k_val,n], axis=0)
            corr = 1 + (3*l**2/(np.linalg.norm(self.r[:,k_val,n], axis=0)**beta*self.c**2))

            acceleration[:,n] = acceleration_sum - self.G*self.M_Sun/np.linalg.norm(self.r[:,k_val,n], axis=0)**beta*corr*unit_r_sun
        return acceleration

    def solver_relativistic(self, beta):
        """Solver relativistic + Verlet"""

        # initalize r and v matrices
        self.r[:,0,:] = self.r0
        self.v[:,0,:] = self.v0

        # size of time step (use as argument instead of T or n?)
        dt = self.ts[1] - self.ts[0]

        for k in range(self.n-1):

            self.k = k  # current index (in time), why self..?

            acceleration1 = self.relativity(k, beta)

            self.r[:,k+1,:]  = self.r[:,k,:] + self.v[:,k,:]*dt + 0.5*acceleration1*dt**2
            acceleration2    = self.relativity(k+1, beta)
            self.v[:,k+1,:]  = self.v[:,k,:] + 0.5*(acceleration1+acceleration2)*dt
        return self.r, self.v, self.ts


    def solve(self, method, beta=2, SunInMotion=False):
        """Solves the system of ODEs


        method : string
             name of desired method, should be 'Euler' or 'Verlet'
        """

        # initalize r and v matrices
        self.r[:,0,:] = self.r0
        self.v[:,0,:] = self.v0

        # size of time step
        dt = self.ts[1] - self.ts[0]


        if SunInMotion == True:

            #center of mass correction
            total_mass = np.sum(self.M)

            R = np.zeros(2)
            V = np.zeros(2)
            Rx = np.sum(self.M*self.r[0,0,:])/total_mass
            Ry = np.sum(self.M*self.r[1,0,:])/total_mass
            Vx = np.sum(self.M*self.v[0,0,:])/total_mass
            Vy = np.sum(self.M*self.v[1,0,:])/total_mass
            R = np.array([Rx, Ry])
            V = np.array([Vx, Vy])

            for i in range(self.Np):
                self.r[:,0,i] -= R
                self.v[:,0,i] -= V

            for k in range(self.n-1):
                self.k = k  # current index (in time)

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
                self.k = k  # current index (in time)

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
