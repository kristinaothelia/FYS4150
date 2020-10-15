"""
FYS4150 - Project 3:
A file that contains various functions used in the project
"""
import os, random, sys, argparse, csv, time

import numpy               as np
import pandas              as pd
import matplotlib.pyplot   as plt

#------------------------------------------------------------------------------

def GetData(filename=''):
    """
    Function for reading csv files
    Input: Filename as a string
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    """
    cwd      = os.getcwd()
    fn       = cwd + filename
    nanDict  = {}
    Data     = pd.read_csv(fn, header=0, skiprows=0, index_col=False, na_values=nanDict)

    return Data

"""
def Grav(G, M, m, r):
    return (G*M*m)/r**2
"""

def get_acceleration(GM, t, pos):
    r_vec = np.array([0, 0]) - pos[t, :]
    r     = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
    acc   = GM*r_vec / r**3

    return acc
'''
def ForwardEuler(G, ts, pos, vel, dt):
    """
    Forwrd Euler method. Returns position and velocity
    """
    start_time = time.time()

    for t in range(ts-1):
        pos[t+1, :] = pos[t, :] + vel[t, :]*dt
        vel[t+1, :] = vel[t, :] + get_acceleration(G, t, pos)*dt

    print("Forward Euler time: ", time.time()-start_time)
    # Trenger kanskje ikke return..?
    return pos, vel

def Verlet(G, ts, pos, vel, acc, dt):
    """
    Verlet method. Returns position and velocity
    """
    start_time = time.time()

    for t in range(ts-1):
        pos[t+1, :] = pos[t, :] + vel[t, :]*dt + 0.5*acc[t, :]*dt**2
        acc[t+1, :] = get_acceleration(G, t+1, pos)
        vel[t+1, :] = vel[t, :] + 0.5*(acc[t, :] + acc[t+1, :])*dt

    print("Verlet time: ", time.time()-start_time)
    # Trenger kanskje ikke return..?
    return pos, vel
'''

def Plot_Sun_Earth_system(pos, label=''):

    plt.plot(pos[:, 0], pos[:, 1], label=label)
    plt.axis("equal")
    plt.legend()
