import numpy as np 
from Solver                 import Solver
from main import find_last_min, angular_momentum, Energy


init_pos = np.array([[1,0]])            # [AU]
init_vel = np.array([[0,2*np.pi]])      # [AU/yr]

init_pos = np.transpose(init_pos)
init_vel = np.transpose(init_vel)

solver = Solver(M_E, init_pos, init_vel, Np, T, int(n[i]))

pos_V, vel_V, t_ V = solver.solve(method = "Euler")



'''
def test_conserved_energy():

    kinetic, potential = Energy(vel, pos, time, title='')

    total_energy  =  U+K
    epsilon       = 1e-5

    assert total_energy <,\
    print('The function failed')
    print('The function...')
'''


'''
def test_conserved_angular_momentum():

    angular_momentum = angular_momentum(vel, pos, time, title='')
    epsilon = 1e-5

    assert index_minimum == solution,\
    print('The function failed')
    print('The function...')
'''


'''
distances = [1, 1, 1, 1, 1, 1, 2, 4, 3, 7, 0.3, 6, 7, 8, 9, 10] # length 16
distances = distances[-9:]     # [4, 3, 7, 0.3, 6, 7, 8, 9, 10] # length 9

def test_find_last_min():

    index_minimum = find_last_min(distances)
    solution      = 3

    assert index_minimum == solution,\
    print('The function failed to find the index of the last minimum')
    print('The function correctly extracted the index of the last minimum')
'''