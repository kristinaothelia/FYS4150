import numpy as np 


from Solver                 import Solver
from SolarSystem            import SolarSystem
from main                   import find_last_min, angular_momentum, Energy
from pytest import approx

planet_names_ = ['Earth', 'Jupiter']
planets_      = SolarSystem(planet_names_)


M_E     = planets_.mass[0]      # [kg]

n       = 5*int(1e4)            # because of computational time

Np = 1
T  = 10

init_pos = np.array([[1,0]])            # [AU]
init_vel = np.array([[0,2*np.pi]])      # [AU/yr]
init_pos = np.transpose(init_pos)
init_vel = np.transpose(init_vel)

solver = Solver(M_E, init_pos, init_vel, Np, T, int(n))
pos_V, vel_V, t_V = solver.solve(method = "Euler")


distances = [1, 1, 1, 1, 1, 1, 2, 4, 3, 7, 0.3, 6, 7, 8, 9, 10] # length 16
distances = distances[-9:]     # [4, 3, 7, 0.3, 6, 7, 8, 9, 10] # length 9


def test_conserved_energy():

    kinetic, potential = Energy(vel_V, pos_V, time=t_V, title='')

    total_energy  = kinetic+potential
    mean_energy   = np.mean(total_energy)
    epsilon       = 5

    energy_max    = np.min(total_energy)
    energy_min    = np.max(total_energy)

    assert np.abs(energy_max - energy_min) < epsilon,\
    print('Energy is not conserved')


def test_conserved_angular_momentum():

    ang_mom  = angular_momentum(vel_V, pos_V, time=t_V, title='')
    epsilon  = 0.5
    min_ang  = np.min(ang_mom)
    max_ang  = np.max(ang_mom)
    #assert min_ang == approx(max_ang, rel=epsilon),\
    assert np.abs(min_ang - max_ang) < epsilon,\
    print('Angular momentum is not conserved')


def test_find_last_min():

    index_minimum = find_last_min(distances)
    solution      = 3

    assert index_minimum == solution,\
    print('The function fails to find index of last minimum')
