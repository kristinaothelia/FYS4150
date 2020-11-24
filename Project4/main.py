"""
Main program for FYS4150 - Project 4

Studies of phase transitions in magnetic systems
"""

import ising_model       as ising
import two_x_two         as I2

import sys, os, time
import argparse
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd


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


# Global parameters
# ??

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    group  = parser.add_mutually_exclusive_group()

    group.add_argument('-1', '--c4', action="store_true", help="Project 4, c)")
    group.add_argument('-2', '--d4',  action="store_true", help="Project 4, d)")


    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args  = parser.parse_args()
    ex_4c = args.c4
    ex_4d = args.d4

    print(parser.description)

    if ex_4c == True:
        print("--------------------------------------------------------------")
        print("2x2 latice")
        print("--------------------------------------------------------------")

        # Initial conditions
        max_cycles = 1e7            # Max. MC cycles
        #max_cycles = 10000000
        L          = 2              # Number of spins
        temp       = 1              # [kT/J] Temperature
        J          = 1

        '''
        # MC
        I2.twoXtwo(L, temp, max_cycles)
        # Analytic solutions
        I2.Analythical_2x2(J, L, temp)
        '''

        # Analytic solutions
        A_E, A_SH, A_Mag, A_Suscept, A_MagAbs = I2.Analythical_2x2(J, L, temp)
        Analyticals = I2.DataFrameSolution(A_E, A_SH, A_Mag, A_Suscept, A_MagAbs)

        log_scale   = np.logspace(2, int(np.log10(max_cycles)), (int(np.log10(max_cycles))-1), endpoint=True)

        error_vs_cycles = False

        if error_vs_cycles:

            # Expand array of max_cycles to run:
            MC_runs   = np.outer(log_scale, [1,5]).flatten() # taking the outer product
            MC_runs   = MC_runs[1:-1] # removing first and last value, maybe not?

            # Numerical solutions
            list_num_dfs = I2.twoXtwo(L, temp, MC_runs)
            Numericals   = pd.concat(list_num_dfs)

            # Get array of MeanMagnetizationAbs for plotting
            numeric_MAbs  = Numericals['MeanMagnetizationAbs'].to_numpy(dtype=np.float64)
            analytic_MAbs = Analyticals['MeanMagnetizationAbs'].to_numpy(dtype=np.float64)

            # Calculating the error (use f.ex. rel. error instead?)
            error = abs(numeric_MAbs-analytic_MAbs)

            plot_MCcycles_vs_err(MC_runs, error)

        else:
            MC_runs  = log_scale # what we initially had

            # Numerical solutions
            list_num_dfs = I2.twoXtwo(L, temp, MC_runs)
            Numericals   = pd.concat(list_num_dfs)

            print('\nTable of Analytical Solutions of 2x2 Ising-Model:','\n'+'-'*49+'\n')
            print(Analyticals)

            print('\n\nTable of Numerical Solutions of 2x2 Ising-Model:','\n'+'-'*48+'\n')
            print(Numericals)


    elif ex_4d == True:
        print("--------------------------------------------------------------")
        print("20x20 latice")
        print("--------------------------------------------------------------")

        # Initial conditions
        max_cycles = 1e7            # Max. MC cycles
        max_cycles = 10000000

        L  = 20      # Number of spins
        T1 = 1.0     # [kT/J] Temperature
        T2 = 2.4     # [kT/J] Temperature

        # MC
        #I2.twoXtwo(L, T1, max_cycles)
        #I2.twoXtwo(L, T2, max_cycles)
