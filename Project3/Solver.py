import os
import pandas as pd
import numpy as np

import functions            as func

class Solver:
    def __init__(self, total_time, dt):

        self.total_time = total_time
        self.dt         = dt
        self.ts         = int(self.total_time/self.dt)        # Time steps


    # get_acceleration maa kanskje inn i SolarSystem?


    def ForwardEuler(G, pos, vel):
        """
        Forwrd Euler method. Returns position and velocity
        """
        start_time = time.time()

        for t in range(self.ts-1):
            pos[t+1, :] = pos[t, :] + vel[t, :]*self.dt
            vel[t+1, :] = vel[t, :] + func.get_acceleration(G, t, pos)*self.dt

        print("Forward Euler time: ", time.time()-start_time)
        # Trenger kanskje ikke return..?
        return self.pos, self.vel   # ???


    def Verlet(G, pos, vel, acc):
        """
        Verlet method. Returns position and velocity
        """
        start_time = time.time()

        for t in range(self.ts-1):
            pos[t+1, :] = pos[t, :] + vel[t, :]*self.dt + 0.5*acc[t, :]*self.dt**2
            acc[t+1, :] = func.get_acceleration(G, t+1, pos)
            vel[t+1, :] = vel[t, :] + 0.5*(acc[t, :] + acc[t+1, :])*self.dt

        print("Verlet time: ", time.time()-start_time)
        # Trenger kanskje ikke return..?
        return self.pos, self.vel   # ???
