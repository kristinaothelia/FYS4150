import numpy as np
import numba


@numba.njit(cache = True)
def initial_energy(spin_matrix, n_spins):
    """
    This function calculates the initial energy and magnetization of the input
    configuration.

    Input:
    spin_matrix | The initial lattice as a matrix
    n_spins     | Number of spins

    Output:
    E           | Initial energy
    M           | Initial magnetization
    """
    E = 0
    M = 0

    for i in range(n_spins):
        for j in range(n_spins):

            left  = spin_matrix[i-1, j] if i>0 else spin_matrix[n_spins - 1, j]
            above = spin_matrix[i, j-1] if j>0 else spin_matrix[i, n_spins - 1]

            E -= spin_matrix[i,j]*(left+above)
            M += spin_matrix[i,j]

    return E, M


@numba.njit(cache=True)
def MC(spin_matrix, n_cycles, temp):
    """

    Monte-Carlo algorithm for solving the Ising model with implemented Markov
    chain and Metropolis algorithm.    stemmer..?

    Monte-Carlo algorithm for calculating the energy, magnetization and their
    suared of a Ising model lattice. The number of accepted configurations
    according to the metropolis algorithm is also tracked.
    Periodic boudary conditions are applied.

    Input:
    spin_matrix | The initial lattice (ordered or random) as a matrix
    num_cycles  | Number of Monte Carlo cycles
    temp        | Temperature

    Output:
    quantities  | A matrix with the energy, magnetization, energy**2,
                  magnetization**2, absolute value of the magnetization and the
                  counter for finding the number of accepted configurations
    """
    n_spins     = len(spin_matrix)
    # Matrix for storing calculated expectation and variance values, five variables
    quantities  = np.zeros((int(n_cycles), 6))
    accepted    = 0

    # Initial energy and magnetization
    E, M        = initial_energy(spin_matrix, n_spins)

    for i in range(1, n_cycles+1):
        for j in range(n_spins*n_spins):

            # Picking a random lattice position
            ix = np.random.randint(n_spins)
            iy = np.random.randint(n_spins)

            # Finding the surrounding spins using periodic boundary conditions
            left  = spin_matrix[ix - 1, iy] if ix > 0 else spin_matrix[n_spins - 1, iy]
            right = spin_matrix[ix + 1, iy] if ix < (n_spins - 1) else spin_matrix[0, iy]
            above = spin_matrix[ix, iy - 1] if iy > 0 else spin_matrix[ix, n_spins - 1]
            below = spin_matrix[ix, iy + 1] if iy < (n_spins - 1) else spin_matrix[ix, 0]

            # Calculating the energy change
            dE = (2 * spin_matrix[ix, iy] * (left + right + above + below))

            # Evaluating the proposet new configuration
            if np.random.random() <= np.exp(-dE / temp):

                # Changing the configuration if accepted
                spin_matrix[ix, iy] *= -1.0
                E                    = E + dE
                M                    = M + 2*spin_matrix[ix, iy]
                accepted            += 1

        # Store values in output matrix
        quantities[i-1,0] += E
        quantities[i-1,1] += M
        quantities[i-1,2] += E**2
        quantities[i-1,3] += M**2
        quantities[i-1,4] += np.abs(M)
        #quantities[i-1,5] += accepted

    return quantities
