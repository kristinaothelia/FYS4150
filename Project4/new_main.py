import sys, os, time, argparse

#import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import plots             as P
import matplotlib.pyplot as plt
import scipy.integrate as integrate

from   numba import jit, njit, prange, set_num_threads

# -----------------------------------------------------------------------------

def Analytical_2x2(J, L, temp):
    """
    Computes the analytical solutions to the 2x2 lattice.
    input:
    - J: binding constant
    - L: dimension of 2D lattice
    - temp: temperature of system
    returns:
    - E: expectation value of energy
    - Cv: specific heat capacity
    - M: expectation value of magnetization
    - X: susceptablity
    - MAbs: expectation value of mean value of magnetization
    """

    const = 8*J/temp

    # the partition function
    Z    = 12 + 2*np.exp(-const) + 2*np.exp(const)

    # expectation values for E
    E_avg       = 16*J*    (np.exp(-const) - np.exp(const)) / Z
    E2_avg      = 128*J**2*(np.exp(-const) + np.exp(const)) / Z
    E_var       = E2_avg - E_avg**2 # 512*J**2 * (Z-6) / Z**2 ??

    # expectation values for M
    M_avg       = 0
    M2_avg      = 32*(1 + np.exp(const)) / Z
    M_abs_avg   = 8*(2 + np.exp(const))  / Z
    M_var       = M2_avg - M_avg**2

    # scaling by L? why is this correct?
    A_Energy            = E_avg / L**2
    A_SpecificHeat      = E_var / (temp**2 * L**2)  # Cv
    A_Magnetization     = M_avg / L**2
    A_MagnetizationAbs  = M_abs_avg / L**2
    A_Susceptibility    = M_var / (temp * L**2)     # X, (32/(4*Z))*(1+np.exp(ang))

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

@njit(cache=True)
def numerical_solution(spin_matrix, n_cycles, temp, L, abs=False):

    # Compute quantities
    quantities, Naccept = MC(spin_matrix, n_cycles, temp)

    E_avg               = np.mean(quantities[:,0])
    M_avg               = np.mean(quantities[:,1])
    E2_avg              = np.mean(quantities[:,2])
    M2_avg              = np.mean(quantities[:,3])
    M_abs_avg           = np.mean(quantities[:,4])

    # variance for E and M
    E_var               = (E2_avg - E_avg**2)/(L**2)
    if abs:
        M_var = (M2_avg - M_abs_avg**2)/(L**2)
    else:
        M_var               = (M2_avg - M_avg**2)/(L**2)

    # scale with L^2
    Energy              = E_avg    /(L**2)
    Magnetization       = M_avg    /(L**2)
    MagnetizationAbs    = M_abs_avg/(L**2)
    SpecificHeat        = E_var    /(temp**2)
    Susceptibility      = M_var    /(temp)

    return Energy, Magnetization, MagnetizationAbs, SpecificHeat, Susceptibility, Naccept

def twoXtwo(L, temp, runs):

    spin_matrix = np.ones((L, L), np.int8)
    list_num_df = []  #what does df mean?

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


            print("hi")#; sys.exit(1)
            Energy, Magnetization, MagnetizationAbs, SpecificHeat, Susceptibility, Nacc \
             = numerical_solution(spin_matrix, n_cycles, temp[t], L, abs=False)


            E[m,t,:]         = Energy
            Mag[m,t,:]       = Magnetization
            MagAbs[m,t,:]    = MagnetizationAbs
            SH[m,t,:]        = SpecificHeat
            Suscept[m,t,:]   = Susceptibility

            Naccept[m,t,:]   = Nacc

    return E, Mag, MagAbs, SH, Suscept, Naccept

def plot_MCcycles_vs_err(mc_cycles, error):
    """Plotting error vs. number of MC cycles.

    loglog or semilog?
    Need better adjustment of plot.
    New title, xlabel, ylabel etc.
    """
    plt.figure(figsize=(15, 10))

    plt.semilogx(mc_cycles, error, 'bo-') # or loglog? semilog, only one axis is logarithmic

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

    #what does this do? remove comments?
    xmin = 0.50e2 #f'{np.min(mc_cycles):10.2e}'  #0.5e2
    xmax = 1.5e7  #f'{np.max(mc_cycles):10.2e}'  #1.1e7
    #plt.ylim(error.min(), error.max()); #plt.ylim(1e-6, 1e-3)
    #plt.xlim(xmin, xmax)

    plt.title('Error of the Mean Abs. Magnetization',fontsize=15)
    plt.xlabel('Number of Monte-Carlo Cycles',fontsize=15)
    plt.ylabel('error',fontsize=15)
    plt.xticks(fontsize=13);plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.savefig(f'results/plots/4c/ErrorMeanMagnetizationAbs')
    plt.show()


def plot_expected_net_mag(L, temp, runs):
    """
    plotting expected net mag

    the plots show that value goes to zero (expected value)
    for large (1e7) number of mc-cycles.

    inspo from rapport (u know who)...

    should probably increase N...?
    not sure what it 'should be' or why...
    """

    colors      = ['rosybrown','lightcoral','indianred','firebrick','darkred','red']
    spin_matrix = np.ones((L, L), np.int8)

    plt.figure(figsize=(10, 6))

    N     = 30  # number of times to run n_cycles
    count = 0

    for n_cycles in runs:

        c     = colors[count]
        count += 1
        for i in range(N):

            E, Mag, MagAbs, SH, Suscept, Naccept = numerical_solution(spin_matrix, int(n_cycles), temp, L)
            plt.semilogx(int(n_cycles), Mag, 'o', color=c)

    plt.title('Spread of Expected Magnetic Field of Matrix', fontsize=15)
    plt.xlabel('Number of Monte-Carlo Cycles', fontsize=15)
    plt.ylabel(r'\langle M \rangle', fontsize=15)
    plt.xticks(fontsize=13);plt.yticks(fontsize=13)
    plt.savefig(f'results/plots/4c/SpreadOfExpectedMagneticField')
    plt.show()





ex_c = False
ex_d = False
ex_e = False
ex_f = True
ex_g = False


#set_num_threads(3)

if ex_c:

    # Initial conditions for Monte Carlo simulation
    max_cycles = 1e7          # Max MC cycles
    L          = 2            # Number of spins
    temp       = 1            # [kT/J] Temperature
    J          = 1            # binding constant


    ###-->
    log_scale = np.logspace(2, int(np.log10(max_cycles)),\
                           (int(np.log10(max_cycles))-1), endpoint=True)
    MC_runs   = np.outer(log_scale, [1,5]).flatten() # taking the outer product
    MC_runs   = MC_runs[1:-1]                        # removing first and last value
    ###<---

    # Analytic solutions
    A_E, A_Cv, A_M, A_X, A_MAbs = Analytical_2x2(J, L, temp)
    Analyticals  = DataFrameSolution(A_E, A_Cv, A_M, A_X, A_MAbs)

    # Numerical solutions
    list_num_dfs = twoXtwo(L, temp, MC_runs)
    Numericals   = pd.concat(list_num_dfs)

    print('\nTable of Analytical Solutions of 2x2 Ising-Model:','\n'+'-'*49+'\n')
    print(Analyticals)
    print('\n\nTable of Numerical Solutions of 2x2 Ising-Model:','\n'+'-'*48+'\n')
    print(Numericals)

    error_vs_cycles        = True  #visualize error as function of MC runs
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
        plot_expected_net_mag(L, temp, runs=log_scale)


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
    MC_runs  = int(1e7)
    new_arrays = False
    if new_arrays:
        print("making npy files, 10^7 MC runs")
        E, Mag, MagAbs, SH, Suscept, n_acc = two_temps(L, MC_runs, temp_arr)
        np.save("E.npy", E)
        np.save("Mag.npy", Mag)
        np.save("MagAbs.npy", MagAbs)
        np.save("SH.npy", SH)
        np.save("Suscept.npy", Suscept)
        np.save("n_acc.npy", n_acc)
        print("file saved!")
        sys.exit(1)

    E = np.load("E.npy")
    SH = np.load("SH.npy")
    lim = int(0.90*MC_runs) #last 10% of data points

    E_T1 = E[0, 1, lim:-1]#*L**2
    #plt.plot(E_T1, 'o-')
    #plt.show()

    plt.hist(E_T1, bins=20, density=True)
    print(E_T1)

    plt.show()


    """
    histo = np.histogram(E_T1, bins=300)
    print(histo[1])
    area = np.abs(integrate.trapz(histo[1]))
    print(area)
    print(histo[1]/area)
    A = histo[1]/area
    var_1 = np.var(A)
    print(var_1)

    Cv_mean = np.mean(SH[0,1,lim:])
    print("Cv = ", Cv_mean)
    kB = 1.38064852e-23
    var_2 = T2**2*Cv_mean*L**2
    print(var_2)
    """


    #Tenkte å prøve og skrive om dette eller noe kanskje..
    #https://github.com/siljeci/FYS4150/blob/master/Project4/CODES/prob.py

if ex_f:
    """
    skriv noe
    """
    L  = [40, 60, 80, 100]          # Number of spins
    T1 = 2.0                        # [kT/J] Temperature
    T2 = 2.3                        # [kT/J] Temperature
    dT = 0.02
    N  = int(round((T2 - T1)/dT))   # nr of steps
    NL = int(len(L))
    T  = np.linspace(T1, T2, N)

    names       = ['Energy','Magnetization','Abs. Magnetization',\
                   'Specific Heat','Susceptibility']

    ylabels      = [r'$\langle E\rangle$', r'$\langle M \rangle$',\
                    r'$\langle|M|\rangle$', r'$C_V$', r'$\chi$']

    save_as      = ['energy','mag','Mabs','CV','CHI']

    MC_runs     = int(1e6)
    stable      = int(0.10*MC_runs)

    E_val       = np.zeros((NL, N))  # rows L, columns N (temperature)
    M_val       = np.zeros_like(E_val); Cv_val    = np.zeros_like(E_val)
    X_val       = np.zeros_like(E_val); M_abs_val = np.zeros_like(E_val)


    for l in range(NL):

        spin_matrix = np.ones((L[l], L[l]), np.int8)
        print("PT for L=", L[l])

        for i in range(N):
            Energy, Magnetization, MagnetizationAbs, SpecificHeat, Susceptibility, Naccept \
             = numerical_solution(spin_matrix, MC_runs, T[i], L[l], abs=True)
            E_val[l,i]      = Energy
            M_val[l,i]      = Magnetization
            M_abs_val[l,i]  = MagnetizationAbs
            Cv_val[l,i]     = SpecificHeat
            X_val[l,i]      = Susceptibility

    # Make and save plots for all metrics, for all L
    vals = [E_val, M_val, M_abs_val, Cv_val, X_val]

    for i in range(len(names)):
        plt.figure()
        val = vals[i]
        for l in range(NL):
            plt.plot(T, val[l,:], label="L=%g" %L[l])

        print("Saving phase transition plot for %s" %names[i])
        P.plot_4f(name=names[i], ylabel=ylabels[i], save_as=save_as[i])
        #plt.show()

if ex_g:
    """
    Find T_C
    """
