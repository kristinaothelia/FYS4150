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
    """
    E and M are int
    """

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
    quantities  = np.zeros((int(n_cycles), 6))  # dtype=np.float64
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

        # update expectation values and store in output matrix
        quantities[i-1,0] += E
        quantities[i-1,1] += M
        quantities[i-1,2] += E**2
        quantities[i-1,3] += M**2
        quantities[i-1,4] += np.abs(M)
        #quantities[i-1,5] += accepted

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

    #Naccept = Naccept*norm
    return Energy, Magnetization, MagnetizationAbs, SpecificHeat, Susceptibility, Naccept

def twoXtwo(L, temp, runs):

    spin_matrix = np.ones((L, L), np.int8)
    list_num_df = []

    for n_cycles in runs:
        Energy, Magnetization, MagnetizationAbs, SpecificHeat, Susceptibility, Naccept = \
        numerical_solution(spin_matrix, n_cycles, temp, L)
        list_num_df.append(DataFrameSolution(Energy, SpecificHeat, Magnetization, Susceptibility, MagnetizationAbs, n_cycles))

    return list_num_df


#@numba.jit(cache=True, parallel=True)
def n_x_n(L, temp1, temp2, runs, ordered=True):

    num_df_T1 = []
    num_df_T2 = []
    #num_accept  = []

    #quantities  = np.zeros((int(n_cycles), 6))  

    if ordered == False:
            spin_matrix = np.random.choice((-1, 1), (L*L)) # random start configuration
    else:
        spin_matrix = np.ones((L, L), np.int8)             # initial state (ground state)

    for n_cycles in runs:

        n_cycles = int(n_cycles)

        E1, Mag1, MagAbs1, SH1, Suscept1, Naccept1 = \
        numerical_solution(spin_matrix, n_cycles, temp1, L)
        num_df_T1.append(DataFrameSolution(E1, SH1, Mag1, Suscept1, MagAbs1, n_cycles))

        E2, Mag2, MagAbs2, SH2, Suscept2, Naccept2 = \
        numerical_solution(spin_matrix, n_cycles, temp2, L)
        num_df_T2.append(DataFrameSolution(E2, SH2, Mag2, Suscept2, MagAbs2, n_cycles))
        
        #num_accept.append(Naccept)
    return num_df_T1, num_df_T2 #, num_accept


#@numba.njit(cache=True, parallel=True)
@numba.njit(cache=True)
def two_temps(ground_spin_mat, random_spin_mat, L, temp1, temp2, runs):
    """
    Calculating the two temps with 2 different start conditions
    """
    
    E1       = np.zeros((2, len(runs)))
    E2       = np.zeros((2, len(runs)))

    Mag1     = np.zeros((2, len(runs)))
    MagAbs1  = np.zeros((2, len(runs)))
    SH1      = np.zeros((2, len(runs)))
    Suscept1 = np.zeros((2, len(runs)))

    Mag2     = np.zeros((2, len(runs)))
    MagAbs2  = np.zeros((2, len(runs)))
    SH2      = np.zeros((2, len(runs)))
    Suscept2 = np.zeros((2, len(runs)))

    for m in range(2):
        for r in range(len(runs)):

            spin_matrix = ground_spin_mat if m==0 else random_spin_mat

            n_cycles  = int(runs[r])

            quantities1, Naccept1 = MC(spin_matrix, n_cycles, temp1)
            quantities2, Naccept2 = MC(spin_matrix, n_cycles, temp2)

            norm                  = 1.0/float(n_cycles)

            E_avg1               = np.sum(quantities1[:,0])*norm
            M_avg1               = np.sum(quantities1[:,1])*norm
            E2_avg1              = np.sum(quantities1[:,2])*norm
            M2_avg1              = np.sum(quantities1[:,3])*norm
            M_abs_avg1           = np.sum(quantities1[:,4])*norm
            E_var1               = (E2_avg1 - E_avg1**2)/(L**2)
            M_var1               = (M2_avg1 - M_avg1**2)/(L**2)
            Energy1              = E_avg1    /(L**2)
            Magnetization1       = M_avg1    /(L**2)
            MagnetizationAbs1    = M_abs_avg1/(L**2)
            SpecificHeat1        = E_var1    /(temp1**2)
            Susceptibility1      = M_var1    /(temp1)   

            E1[m,r]=Energy1
            Mag1[m,r]=Magnetization1
            MagAbs1[m,r]=MagnetizationAbs1
            SH1[m,r]=SpecificHeat1
            Suscept1[m,r]=Susceptibility1

            E_avg2               = np.sum(quantities2[:,0])*norm
            M_avg2               = np.sum(quantities2[:,1])*norm
            E2_avg2              = np.sum(quantities2[:,2])*norm
            M2_avg2              = np.sum(quantities2[:,3])*norm
            M_abs_avg2           = np.sum(quantities2[:,4])*norm
            E_var2               = (E2_avg2 - E_avg2**2)/(L**2)
            M_var2               = (M2_avg2 - M_avg2**2)/(L**2)
            Energy2              = E_avg2    /(L**2)
            Magnetization2       = M_avg2    /(L**2)
            MagnetizationAbs2    = M_abs_avg2/(L**2)
            SpecificHeat2        = E_var2    /(temp2**2)
            Susceptibility2      = M_var2    /(temp2) 

            E2[m,r]=Energy2
            Mag2[m,r]=Magnetization2
            MagAbs2[m,r]=MagnetizationAbs2
            SH2[m,r]=SpecificHeat2
            Suscept2[m,r]=Susceptibility2
        
    return E1,E2,Mag1,Mag2,MagAbs1,MagAbs2,SH1,SH2,Suscept1,Suscept2

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

    N     = 10  # number of times to run each 'max-cycles' in runs-array
    count = 0

    for n_cycles in runs:

        c     = colors[count]
        count += 1

        for i in range(N):

            E, Mag, MagAbs, SH, Suscept, Naccept = numerical_solution(spin_matrix, int(n_cycles), temp, L)

            plt.semilogx(int(n_cycles), Mag, 'o', color=c)
    
    plt.title('Spread of expected magnetic field of matrix\
        \n(averaged over MCcycles??, normalized to L**2??)', fontsize=15)
    plt.xlabel('Number of MC cycles', fontsize=15)
    plt.ylabel('<M>', fontsize=15)
    plt.xticks(fontsize=12);plt.yticks(fontsize=12)
    plt.show()
    sys.exit()

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


def expectation_vals(T1, T2, Num_T1, Num_T2, MC_runs):
    """Expectation values as function of Monte Carlo Cycles.

    Hm, mulig inne på noe, men ikke alle ser helt riktig ut.. 
    Burde vel loope over flere verdier/runs/max-cycles..
    Mulig jeg har blandet sammen hva som heter hva.. 
    """

    col     = Numericals_T1.columns.values
    name    = ['Energy','Specific Heat','Magnetization','Susceptibility','Abs. Magnetization']
    
    ylabels = [r'<$\epsilon$>', 'CP', 'M' , r'$\chi$', 'MAbs']
    ydict   = dict(zip(col,ylabels))

    for key, value in zip(col, name):

        ylab   = ydict[key]    # getting ylabel
        val1_o = Num_T1[key]   # values T1 ordered
        val2_o = Num_T2[key]   # values T2 ordered

        plt.semilogx(MC_runs, val1_o, label=f'T={T1} (order)', color='tab:blue')
        plt.semilogx(MC_runs, val2_o, label=f'T={T2} (order)', color='tab:red')

        plt.title(f'Expectation of {value} vs. MC Cycles', fontsize=15)
        plt.xlabel('Number of MC cycles', fontsize=15)
        plt.ylabel(f'{ylab}', fontsize=15)
        plt.xticks(fontsize=12);plt.yticks(fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()


def expected_vals_two_temp(MC_runs, T1, T2, val1, val2, name, ylab):
    """
    Function for plotting expectation values vs. MC cycles,
    - two temperatures in two different initial states.
    """

    plt.semilogx(MC_runs, val1[0,:], label=f'T={T1} (order)', color='tab:blue')
    plt.semilogx(MC_runs, val2[0,:], label=f'T={T2} (order)', color='tab:red')

    plt.semilogx(MC_runs, val1[1,:], '--', label=f'T={T1} (disorder)',color='tab:blue')
    plt.semilogx(MC_runs, val2[1,:], '--', label=f'T={T2} (disorder)',color='tab:red')
    
    plt.title(f'Expectation of {name} vs. MC Cycles', fontsize=15)
    plt.xlabel('Number of MC cycles', fontsize=15)
    plt.ylabel(f'{ylab}', fontsize=15)
    plt.xticks(fontsize=12);plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()


ex_c = False
ex_d = False

#ex_c = True
ex_d = True

# Initial conditions
max_cycles = 1e7          # Max. MC cycles
max_cycles = 10000000


if ex_c: 
    L          = 2            # Number of spins
    temp       = 1            # [kT/J] Temperature
    J          = 1

    log_scale = np.logspace(2, int(np.log10(max_cycles)), (int(np.log10(max_cycles))-1), endpoint=True)
    MC_runs   = np.outer(log_scale, [1,5]).flatten() # taking the outer product
    MC_runs   = MC_runs[1:-1] # removing first and last value 
    
    #print(log_scale);print(MC_runs);print(len(MC_runs))

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

    expected_net_magnetism = True

    if error_vs_cycles:

        # Get array of MeanMagnetizationAbs for plotting
        numeric_MAbs  = Numericals['MeanMagnetizationAbs'].to_numpy(dtype=np.float64)
        analytic_MAbs = Analyticals['MeanMagnetizationAbs'].to_numpy(dtype=np.float64)

        # Calculating the error (use f.ex. rel. error instead?)
        error = abs(numeric_MAbs-analytic_MAbs)

        plot_MCcycles_vs_err(MC_runs, error)

    if expected_net_magnetism:

        plot_expected_net_mag(L, temp, runs=log_scale)


if ex_d:
    L  = 20    # Number of spins
    T1 = 1.0   # [kT/J] Temperature
    T2 = 2.4   # [kT/J] Temperature

    log_scale = [1.0,10.0,100.0,1000.0]
    #log_scale = np.logspace(0, int(np.log10(max_cycles)-2), (int(np.log10(max_cycles))-1), endpoint=True)
    multip    = np.arange(1, 10, 0.5) #np.arange(2, 11, 2)   # np.arange(1,6,4) #
    MC_runs   = np.outer(log_scale, multip).flatten() # taking the outer product
    #MC_runs   = MC_runs[3:-1]
    
    print(log_scale);print(multip);print(MC_runs);print(len(MC_runs))

    ordered = False

    s_mat_random = np.ones((L,L), np.int8)
    for s in range(len(s_mat_random)):
        rint = np.random.randint(-1,1)
        if rint == -1:
            s_mat_random[s] *= -1

    # this crashes with numba i think...
    #spin_matrix = np.random.choice((-1, 1), (L*L)) # random start configuration
    
    s_mat_ground = np.ones((L, L), np.int8)         # initial state (ground state)

    start=time.time()
    E1,E2,Mag1,Mag2,MagAbs1,MagAbs2,SH1,SH2,Suscept1,Suscept2 = two_temps(s_mat_ground, s_mat_random, L, T1, T2, MC_runs)
    end=time.time()
    print(end-start)

    list_expect_vals = [[E1,E2],[Mag1,Mag2],[MagAbs1,MagAbs2],[SH1,SH2],[Suscept1,Suscept2]]
    print(len(list_expect_vals))

    names   = ['Energy','Magnetization','Abs. Magnetization','Specific Heat','Susceptibility']
    ylabels = [r'<$\epsilon$>', 'M', 'MAbs', 'CP', r'$\chi$',]

    # the names may be mixed up, havent checked yet..

    for v in range(5):

        val1 = list_expect_vals[v][0]
        val2 = list_expect_vals[v][1]

        expected_vals_two_temp(MC_runs, T1, T2, val1, val2, names[v], ylabels[v])
        plt.show()


    '''
    plt.semilogx(MC_runs, E1[0,:], label=f'T={T1} (order)', color='tab:blue')
    plt.semilogx(MC_runs, E2[0,:], label=f'T={T2} (order)', color='tab:red')
    plt.semilogx(MC_runs, E1[1,:], '--', label=f'T={T1} (disorder)',color='tab:blue')
    plt.semilogx(MC_runs, E2[1,:], '--', label=f'T={T2} (disorder)',color='tab:red')
    plt.title('Energy')
    plt.legend()
    plt.show()

    plt.semilogx(MC_runs, Mag1[0,:], label=f'T={T1} (order)', color='tab:blue')
    plt.semilogx(MC_runs, Mag2[0,:], label=f'T={T2} (order)', color='tab:red')
    plt.semilogx(MC_runs, Mag1[1,:], '--', label=f'T={T1} (disorder)', color='tab:blue')
    plt.semilogx(MC_runs, Mag2[1,:], '--', label=f'T={T2} (disorder)', color='tab:red')
    plt.title('M')
    plt.legend()
    plt.show()

    plt.semilogx(MC_runs, MagAbs1[0,:], label=f'T={T1} (order)', color='tab:blue')
    plt.semilogx(MC_runs, MagAbs2[0,:], label=f'T={T2} (order)', color='tab:red')
    plt.semilogx(MC_runs, MagAbs1[1,:], '--', label=f'T={T1} (disorder)', color='tab:blue')
    plt.semilogx(MC_runs, MagAbs2[1,:], '--', label=f'T={T2} (disorder)', color='tab:red')
    plt.title('Abs M')
    plt.legend()
    plt.show()

    plt.semilogx(MC_runs, SH1[0,:], label=f'T={T1} (order)', color='tab:blue')
    plt.semilogx(MC_runs, SH2[0,:], label=f'T={T2} (order)', color='tab:red')
    plt.semilogx(MC_runs, SH1[1,:], '--', label=f'T={T1} (disorder)', color='tab:blue')
    plt.semilogx(MC_runs, SH2[1,:], '--', label=f'T={T2} (disorder)', color='tab:red')
    plt.title('Specific Heat')
    plt.legend()
    plt.show()

    plt.semilogx(MC_runs, Suscept1[0,:], label=f'T={T1} (order)', color='tab:blue')
    plt.semilogx(MC_runs, Suscept2[0,:], label=f'T={T2} (order)', color='tab:red')
    plt.semilogx(MC_runs, Suscept1[1,:], '--', label=f'T={T1} (disorder)', color='tab:blue')
    plt.semilogx(MC_runs, Suscept2[1,:], '--', label=f'T={T2} (disorder)', color='tab:red')
    plt.title('Susceptibility')
    plt.legend()
    plt.show()
    '''

    ############################################################################


    '''
    numeric_df_T1, numeric_df_T2 = n_x_n(L, T1, T2, MC_runs)
    Numericals_T1 = pd.concat(numeric_df_T1)
    Numericals_T2 = pd.concat(numeric_df_T2)

    
    print(Numericals_T1.dtypes)
    print(Numericals_T1.__sizeof__())
    print(sys.getsizeof(Numericals_T1['MeanEnergy'].values))
    print(Numericals_T1['MeanEnergy'].values.nbytes)
    print(Numericals_T1.memory_usage(deep=True))
    print(Numericals_T1.memory_usage(deep=False))
    print(Numericals_T1._is_copy)
    print(Numericals_T1._is_view)
    sys.exit()
    
    print(Numericals_T1);print('')
    print(Numericals_T2);print('')
    expectation_vals(T1, T2, Numericals_T1, Numericals_T2, MC_runs)
    '''

    ############################################################################

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
