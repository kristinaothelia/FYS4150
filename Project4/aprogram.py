"""
All code in one file.

Just a program for trying out stuff.
"""

# trying out some documentation things
# https://stackoverflow.com/questions/1523427/what-is-the-common-header-format-of-python-files
__author__  = "Anna Eliassen"
__version__ = "3.7.4"


import sys, os, time, numba, argparse

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

import doctest
#doctest.testmod() # must be placed somewhere below the function (f.ex. after name=main)


print(__doc__);print('')

print(f"{'Author(s)':15s}: {__author__:>13s}") 
print(f"{'Python Version':15s}: {__version__:>13s}");print('\n')

#print(f"{'Author':{len(__author__)+3}s}{'Python Version':>14s}",'\n'+'-'*30)
#print(f"{__author__:{len(__author__)+3}s}{__version__:>14s}")

def a_func(x):
    """Illustration of Numpy style docstrings.

    A test function showing the typical 
    layout of "NumPy style" docstrings.
    This docstring includes an example of a doctest.

    More details and other section descriptions at
    https://numpydoc.readthedocs.io/en/latest/format.html

    To print the documentation: print(a_func.__doc__)

    Parameters
    ----------
    x : int
        description of the input parameter `x`.

    Returns
    -------
    y : int
        description of the returned value.

    Examples
    --------
    >>> a_func(5)
    25

    """

    y = x**2
    return y

def Analythical_2x2(J, L, temp):

    ang  = 8*J/temp
    Z    = 12 + 2*np.exp(-ang) + 2*np.exp(ang)

    E_avg       = 16*J*    (np.exp(-ang) - np.exp(ang)) / Z
    E2_avg      = 128*J**2*(np.exp(-ang) + np.exp(ang)) / Z
    E_var       = E2_avg - E_avg**2

    M_avg       = 0
    M2_avg      = 32*(1 + np.exp(ang)) / Z
    M_abs_avg   = 8*(2 + np.exp(ang))  / Z
    M_var       = M2_avg - M_avg**2

    A_Energy            = E_avg / L**2
    A_SpecificHeat      = E_var / (temp**2 * L**2)  # C_V
    A_Magnetization     = M_avg / L**2
    A_MagnetizationAbs  = M_abs_avg / L**2
    A_Susceptibility    = M_var / (temp * L**2)     # Chi, (32/(4*Z))*(1+np.exp(ang))

    '''
    print("\nAnalytical solutions:")
    print("Mean energy:             %f" %A_Energy)
    print("Specific heat:           %f" %A_SpecificHeat)
    print("Mean Magenetization:     %f" %A_Magnetization)
    print("Susceptibility:          %f" %A_Susceptibility)
    print("Mean abs. Magnetization: %f" %A_MagnetizationAbs)
    '''

    return A_Energy, A_SpecificHeat, A_Magnetization, A_Susceptibility, A_MagnetizationAbs

def DataFrameSolution(E, CP, M, CHI, MAbs, N=None):
    """Function creating dataframe with solution values.

    Parameters
    ----------
    N : None, float
        N is None  - used for analytical dataframe
        N is float - number of MC cycles used in numerical calc. 

    TODO
    ----
        Planning to make it more general,
        maybe passing list of key names and list with values,
        or using method argument.
    """

    if N == None:
        solutions = [{'MeanEnergy' : E,\
                      'SpecificHeat' : CP,\
                      'MeanMagnetization' : M,\
                      'Susceptibility' : CHI,\
                      'MeanMagnetizationAbs' : MAbs}]

        dataframe = pd.DataFrame(solutions)
    else:
        solutions = [{'MCcycles' : N,\
                      'MeanEnergy' : E,\
                      'SpecificHeat' : CP,\
                      'MeanMagnetization' : M,\
                      'Susceptibility' : CHI,\
                      'MeanMagnetizationAbs' : MAbs}]

        dataframe = pd.DataFrame(solutions)
        dataframe.set_index('MCcycles', inplace=True)

    return dataframe

@numba.njit(cache = True)
def initial_energy(spin_matrix, n_spins):

    E = 0; M = 0

    for i in range(n_spins):
        for j in range(n_spins):

            left  = spin_matrix[i-1, j] if i>0 else spin_matrix[n_spins - 1, j]
            above = spin_matrix[i, j-1] if j>0 else spin_matrix[i, n_spins - 1]

            E -= spin_matrix[i,j]*(left+above)
            M += spin_matrix[i,j]

    return E, M


@numba.njit(cache=True)
def MC(spin_matrix, n_cycles, temp):

    n_spins     = len(spin_matrix)
    # Matrix for storing calculated expectation and variance values, five variables
    quantities  = np.zeros((int(n_cycles), 6))
    accepted    = 0

    # Initial energy and magnetization
    E, M        = initial_energy(spin_matrix, n_spins)

    for i in range(1, n_cycles+1):
        for j in range(n_spins*n_spins):

            # Picking a random lattice position
            ix = np.random.randint(n_spins)      # random int (0-n_spins), dont include n_spins
            iy = np.random.randint(n_spins)      # random int (0-n_spins), dont include n_spins

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

def numerical_solution(spin_matrix, n_cycles, temp, L):

    # Compute quantities
    quantities       = MC(spin_matrix, n_cycles, temp)

    norm             = 1.0/float(n_cycles)
    E_avg            = np.sum(quantities[:,0])*norm
    M_avg            = np.sum(quantities[:,1])*norm
    E2_avg           = np.sum(quantities[:,2])*norm
    M2_avg           = np.sum(quantities[:,3])*norm
    M_abs_avg        = np.sum(quantities[:,4])*norm

    E_var            = (E2_avg - E_avg**2)/(L**2)
    M_var            = (M2_avg - M_avg**2)/(L**2)

    Energy           = E_avg    /(L**2)
    Magnetization    = M_avg    /(L**2)
    MagnetizationAbs = M_abs_avg/(L**2)
    SpecificHeat     = E_var    /(temp**2)  # * L**2?, no because E_var already /L**2
    Susceptibility   = M_var    /(temp)     # * L**2?

    return Energy, Magnetization, MagnetizationAbs, SpecificHeat, Susceptibility

def twoXtwo(L, temp, runs):

    spin_matrix = np.ones((L, L), np.int8)
    list_num_df = []

    for n_cycles in runs:
        Energy, Magnetization, MagnetizationAbs, SpecificHeat, Susceptibility = \
        numerical_solution(spin_matrix, n_cycles, temp, L)
        list_num_df.append(DataFrameSolution(Energy, SpecificHeat, Magnetization, Susceptibility, MagnetizationAbs, n_cycles))

    return list_num_df


def plot_MCcycles_vs_err(mc_cycles, error):
    """Plotting error vs. number of MC cycles.

    Need better adjustment of plot.
    New title, xlabel, ylabel etc.
    """
    plt.figure(figsize=(7, 4))   # plot the calculated values, (12, 7)
    
    #plt.loglog(mc_cycles, error, 'bo-')
    plt.semilogx(mc_cycles, error, 'bo-') # or loglog?

    # zip joins x and y coordinates in pairs
    for x,y in zip(mc_cycles,error):

        label = f'{y:10.2e}'

        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-10), # distance from text to points (x,y)
                     color='black',
                     weight='bold',
                     size='smaller',
                     rotation='0',   # plot seems weird w/angle other than 0 or 360..?
                     va='top',       #  [ 'center' | 'top' | 'bottom' | 'baseline' ]
                     ha='right')     #  [ 'left' | 'right' | 'center']

    xmin = 0.50e2 #f'{np.min(mc_cycles):10.2e}'  #0.5e2
    xmax = 1.5e7  #f'{np.max(mc_cycles):10.2e}'  #1.1e7

    #plt.ylim(error.min(), error.max()); #plt.ylim(1e-6, 1e-3)
    #plt.xlim(xmin, xmax)

    plt.title('Error vs. Number of Monte-Carlo Cycles',fontsize=15)
    plt.xlabel('N cycles',fontsize=15)
    plt.ylabel('error',fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

ex_c = False
ex_d = False

ex_c = True
#ex_d = True

# Initial conditions
max_cycles = 1e7          # Max. MC cycles
max_cycles = 10000000


if ex_c: 
    L          = 2            # Number of spins
    temp       = 1            # [kT/J] Temperature
    J          = 1

    log_scale = np.logspace(2, int(np.log10(max_cycles)), (int(np.log10(max_cycles))-1), endpoint=True)
              # = np.logspace(2, 7, 6, endpoint=True)
              # = [1.e+02 1.e+03 1.e+04 1.e+05 1.e+06 1.e+07]
    MC_runs   = np.outer(log_scale, [1,5]).flatten() # taking the outer product
              # [1.e+02 5.e+02 1.e+03 5.e+03 1.e+04 5.e+04 1.e+05 5.e+05 1.e+06 5.e+06 1.e+07 5.e+07]
    MC_runs   = MC_runs[1:-1] # removing first and last value 
    
    print(log_scale)
    print(MC_runs)
    print(len(MC_runs))

    # Analytic solutions
    A_E, A_SH, A_Mag, A_Suscept, A_MagAbs = Analythical_2x2(J, L, temp)
    Analyticals  = DataFrameSolution(A_E, A_SH, A_Mag, A_Suscept, A_MagAbs)

    # Numerical solutions
    list_num_dfs = twoXtwo(L, temp, MC_runs)
    Numericals   = pd.concat(list_num_dfs)

    print('\nTable of Analytical Solutions of 2x2 Ising-Model:','\n'+'-'*49+'\n')
    print(Analyticals)

    print('\n\nTable of Numerical Solutions of 2x2 Ising-Model:','\n'+'-'*48+'\n')
    print(Numericals)

    error_vs_cycles = True

    if error_vs_cycles:

        # Get array of MeanMagnetizationAbs for plotting
        numeric_MAbs  = Numericals['MeanMagnetizationAbs'].to_numpy(dtype=np.float64)
        analytic_MAbs = Analyticals['MeanMagnetizationAbs'].to_numpy(dtype=np.float64)

        # Calculating the error (use f.ex. rel. error instead?)
        error = abs(numeric_MAbs-analytic_MAbs)

        plot_MCcycles_vs_err(MC_runs, error)

if ex_d:
    L  = 20      # Number of spins
    T1 = 1.0     # [kT/J] Temperature
    T2 = 2.4     # [kT/J] Temperature

    # MC
    '''
    start = time.time()
    twoXtwo(L, T1, max_cycles)
    end   = time.time()
    print('time:', (end-start))   # 201.3737907409668
    '''
    
    '''
    start = time.time()
    twoXtwo(L, T2, max_cycles)
    end = time.time()
    print('time:', (end-start))   # 225.60117769241333
    '''
