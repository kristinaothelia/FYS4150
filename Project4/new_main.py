import sys, os, time, argparse

#import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import plots             as P

from   numba import jit, njit, prange, set_num_threads

# -----------------------------------------------------------------------------

def Analythical_2x2(J, L, temp):

    ang  = 8*J/temp
    Z    = 12 + 2*np.exp(-ang) + 2*np.exp(ang)

    E_avg       = 16*J*    (np.exp(-ang) - np.exp(ang)) / Z
    E2_avg      = 128*J**2*(np.exp(-ang) + np.exp(ang)) / Z
    E_var       = E2_avg - E_avg**2 # 512*J**2 * (Z-6) / Z**2 ??

    M_avg       = 0
    M2_avg      = 32*(1 + np.exp(ang)) / Z
    M_abs_avg   = 8*(2 + np.exp(ang))  / Z
    M_var       = M2_avg - M_avg**2

    A_Energy            = E_avg / L**2
    A_SpecificHeat      = E_var / (temp**2 * L**2)  # C_V
    A_Magnetization     = M_avg / L**2
    A_MagnetizationAbs  = M_abs_avg / L**2
    A_Susceptibility    = M_var / (temp * L**2)     # Chi, (32/(4*Z))*(1+np.exp(ang))

    return A_Energy, A_SpecificHeat, A_Magnetization, A_Susceptibility, A_MagnetizationAbs


def DataFrameSolution(E, CP, M, CHI, MAbs, N=None):
    """Function creating dataframe with solution values.

    Parameters
    ----------
    N : None, float
        N is None  - used for analytical dataframe
        N is float - number of MC cycles used in numerical calc.

    Returns
    -------
    DataFrame object
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

@njit(cache=True)
def initial_energy(spin_matrix, n_spins):
    """
    E and M are int
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


@njit(cache=True)
def MC(spin_matrix, n_cycles, temp):

    n_spins     = len(spin_matrix)

    # Matrix for storing calculated expectation and variance values, five variables
    quantities  = np.zeros((int(n_cycles), 6))  # dtype=np.float64
    accepted    = np.zeros(int(n_cycles))

    # Initial energy and magnetization
    E, M        = initial_energy(spin_matrix, n_spins)

    for i in range(1, n_cycles+1):
        for j in range(n_spins**2):

            # Picking a random lattice position
            ix = np.random.randint(n_spins)  # dont include n_spins
            iy = np.random.randint(n_spins)  # dont include n_spins

            # Finding the surrounding spins using periodic boundary conditions
            left  = spin_matrix[ix - 1, iy] if ix > 0 else spin_matrix[n_spins - 1, iy]
            right = spin_matrix[ix + 1, iy] if ix < (n_spins - 1) else spin_matrix[0, iy]
            above = spin_matrix[ix, iy - 1] if iy > 0 else spin_matrix[ix, n_spins - 1]
            below = spin_matrix[ix, iy + 1] if iy < (n_spins - 1) else spin_matrix[ix, 0]

            # Calculating the energy change
            dE = (2 * spin_matrix[ix, iy] * (left + right + above + below))

            # Evaluating the proposet new configuration
            if np.random.random() <= np.exp(-dE/temp):
                # Changing the configuration if accepted
                spin_matrix[ix, iy] *= -1.0  #flip spin
                E                   += dE
                M                   += 2*spin_matrix[ix, iy]
                accepted[i]         += 1

        # update expectation values and store in output matrix
        quantities[i-1,0] += E
        quantities[i-1,1] += M
        quantities[i-1,2] += E**2
        quantities[i-1,3] += M**2
        quantities[i-1,4] += np.abs(M)

    return quantities, accepted

def numerical_solution(spin_matrix, n_cycles, temp, L):

    # Compute quantities
    quantities, Naccept = MC(spin_matrix, n_cycles, temp)

    norm                = 1.0/float(n_cycles)
    E_avg               = np.sum(quantities[:,0])*norm
    M_avg               = np.sum(quantities[:,1])*norm
    E2_avg              = np.sum(quantities[:,2])*norm
    M2_avg              = np.sum(quantities[:,3])*norm
    M_abs_avg           = np.sum(quantities[:,4])*norm

    E_var               = (E2_avg - E_avg**2)/(L**2)
    M_var               = (M2_avg - M_avg**2)/(L**2)

    Energy              = E_avg    /(L**2)
    Magnetization       = M_avg    /(L**2)
    MagnetizationAbs    = M_abs_avg/(L**2)
    SpecificHeat        = E_var    /(temp**2)  # * L**2?, no because E_var already /L**2
    Susceptibility      = M_var    /(temp)     # * L**2?

    return Energy, Magnetization, MagnetizationAbs, SpecificHeat, Susceptibility, Naccept

def twoXtwo(L, temp, runs):

    spin_matrix = np.ones((L, L), np.int8)
    list_num_df = []

    for n_cycles in runs:
        Energy, Magnetization, MagnetizationAbs, SpecificHeat, Susceptibility, Naccept = \
        numerical_solution(spin_matrix, n_cycles, temp, L)
        list_num_df.append(DataFrameSolution(Energy, SpecificHeat, Magnetization,\
                                             Susceptibility, MagnetizationAbs, n_cycles))
    return list_num_df


@njit(cache=True)
def two_temps(L, n_cycles, temp):
    """
    Calculating the two temps with 2 different start conditions
    """

    E       = np.zeros((2, len(temp), n_cycles))
    Mag     = np.zeros((2, len(temp), n_cycles))
    MagAbs  = np.zeros((2, len(temp), n_cycles))
    SH      = np.zeros((2, len(temp), n_cycles))
    Suscept = np.zeros((2, len(temp), n_cycles))

    Naccept = np.zeros((2, len(temp), n_cycles))

    ground_spin_mat = np.ones((L, L), np.int8)   # initial state (ground state)

    for m in range(2):
        for t in range(len(temp)):

            #m=0 is ground state, all spin-up
            #m=1 is random state
            if m==0:
                spin_matrix = ground_spin_mat
            else:
                s_mat_random = np.ones((L, L), np.int8)   # a random spin orientation
                #generate spin matrix of ones, then random indices are given 1 og -1sys
                for sw in range(len(s_mat_random)):
                    for sl in range(len(s_mat_random)):
                        rint = np.random.randint(-1,1)
                        if rint == -1:
                            s_mat_random[sw,sl] *= -1
                spin_matrix = s_mat_random


            quantities, Nacc = MC(spin_matrix, n_cycles, temp[t])

            norm       = 1.0/np.arange(1, n_cycles+1)

            E_avg      = np.cumsum(quantities[:,0])*norm
            M_avg      = np.cumsum(quantities[:,1])*norm
            E2_avg     = np.cumsum(quantities[:,2])*norm
            M2_avg     = np.cumsum(quantities[:,3])*norm
            M_abs_avg  = np.cumsum(quantities[:,4])*norm

            E_var            = (E2_avg - E_avg**2)/(L**2)
            M_var            = (M2_avg - M_avg**2)/(L**2)
            Energy           = E_avg    /(L**2)
            Magnetization    = M_avg    /(L**2)
            MagnetizationAbs = M_abs_avg/(L**2)
            #skal ogsaa deles paa L^2?
            SpecificHeat     = E_var    /(temp[t]**2)
            Susceptibility   = M_var    /(temp[t])

            E[m,t,:]         = Energy
            Mag[m,t,:]       = Magnetization
            MagAbs[m,t,:]    = MagnetizationAbs
            SH[m,t,:]        = SpecificHeat
            Suscept[m,t,:]   = Susceptibility

            Naccept[m,t,:]   = Nacc

    return E, Mag, MagAbs, SH, Suscept, Naccept





ex_c = False
ex_d = True
ex_e = False
ex_f = False
ex_g = False


#set_num_threads(3)

if ex_c:

    # Initial conditions
    max_cycles = 1e7          # Max. MC cycles
    max_cycles = 10000000
    L          = 2            # Number of spins
    temp       = 1            # [kT/J] Temperature
    J          = 1

    log_scale = np.logspace(2, int(np.log10(max_cycles)),\
                           (int(np.log10(max_cycles))-1), endpoint=True)
    MC_runs   = np.outer(log_scale, [1,5]).flatten() # taking the outer product
    MC_runs   = MC_runs[1:-1]                        # removing first and last value

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

    error_vs_cycles        = False
    expected_net_magnetism = False

    if error_vs_cycles:

        # Get array of MeanMagnetizationAbs for plotting
        numeric_MAbs  = Numericals['MeanMagnetizationAbs'].to_numpy(dtype=np.float64)
        analytic_MAbs = Analyticals['MeanMagnetizationAbs'].to_numpy(dtype=np.float64)

        # Calculating the error (use f.ex. rel. error instead?)
        error = abs(numeric_MAbs-analytic_MAbs)
        plot_MCcycles_vs_err(MC_runs, error)

    if expected_net_magnetism:

        # Plotting expected mean magnetism
        P.plot_expected_net_mag(L, temp, runs=log_scale)


if ex_d:

    L  = 20    # Number of spins
    T1 = 1.0   # [kT/J] Temperature
    T2 = 2.4   # [kT/J] Temperature

    temp_arr = np.array([T1, T2])
    MC_runs  = int(1e6)

    E, Mag, MagAbs, SH, Suscept, n_acc = two_temps(L, MC_runs, temp_arr)

    P.plot_n_accepted(MC_runs, n_acc, T1, T2)

    expecteds = [E, Mag, MagAbs, SH, Suscept]
    P.expected_vals_two_temp(MC_runs, T1, T2, expecteds)

if ex_e:
    """
    Partition function:
    It is a sum over the two possible spin values for each spon, either up +1 or down −1.
    """
    L  = 20    # Number of spins
    T1 = 1.0   # [kT/J] Temperature
    T2 = 2.4   # [kT/J] Temperature

    temp_arr = np.array([T1, T2])
    MC_runs  = int(1e6)


    #Tenkte å prøve og skrive om dette eller noe kanskje..
    #https://github.com/siljeci/FYS4150/blob/master/Project4/CODES/prob.py

if ex_f:
    """
    skriv noe
    """
    L  = [40, 60, 80, 100]    # Number of spins
    T1 = 2.0   # [kT/J] Temperature
    T2 = 2.3   # [kT/J] Temperature
    dT = 0.05

if ex_g:
    """
    Find T_C
    """
