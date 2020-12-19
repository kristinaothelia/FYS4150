"""
------------------------------------------------------------------------
FYS4150 Computational Physics: Project 5
Disease Modelling using Runge-Kutta 4 and Monte Carlo (on SIRS model)
------------------------------------------------------------------------
"""

# Note: You may need to pip install dataframe_image

import sys, os, time, argparse

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import RK4_MonteCarlo    as Solver
import plots             as P
import dataframe_image   as dfi

from numba import njit

# ----------------------------------------------------------------------------

# Global parameters

a       = 4                     # Rate of transmission
c       = 0.5                   # Rate of immunity loss
bA      = 1                     # Rate of recovery for population A
bB      = 2                     # Rate of recovery for population B
bC      = 3                     # Rate of recovery for population C
bD      = 4                     # Rate of recovery for population D
b_list  = [bA, bB, bC, bD]
pop     = ['A', 'B', 'C', 'D']  # Titles for plots

T   = 12            # Time
N   = 400           # Nr of individuals in population
S_0 = 300           # Initial number of susceptible
I_0 = 100           # Initial number of infected
R_0 = 0             # Initial number of recovered

# Functions to make mean/std table in ex. b)
def magnify():
    return [dict(selector="th",
                 props=[('text-align', 'center'), ("font-size", "13pt")])]

def DataFrameSolution(S_arr, I_arr, R_arr):
    """Function creating a table with solution values.

    This function take in 3 lists of equal length,
    where each list contains the mean value + standard deviation
    of susceptibles, infected and recovered for different populations.
    The function creates and returns a dataframe.

    Parameters
    ----------
    S_arr : list
    I_arr : list
    R_arr : list

    Returns
    -------
    DataFrame object
    """

    groups = ['S', 'I', 'R']

    # creating dataframe and adding style
    dataframe = pd.DataFrame([S_arr, I_arr, R_arr], columns=['A', 'B', 'C', 'D'], index=groups)
    dataframe = dataframe.style.set_properties(**{'text-align':'center'})\
                               .set_properties(subset = pd.IndexSlice[['S'], :],\
                                                **{'color': '#4169E1',\
                                                   'border-color': 'white',
                                                   'font-size' : '13pt'})\
                               .set_properties(subset = pd.IndexSlice[['I'], :],\
                                               **{'color': '#B22222',\
                                                  'border-color': 'white',
                                                  'font-size' : '13pt'})\
                               .set_properties(subset = pd.IndexSlice[['R'], :],\
                                               **{'color': '#228B22',\
                                                  'border-color': 'white',
                                                  'font-size' : '13pt'})\
                               .set_table_styles(magnify())
    return dataframe



if __name__ == '__main__':

    parser    = argparse.ArgumentParser(description=__doc__)
    exercise  = parser.add_mutually_exclusive_group()

    exercise.add_argument('-a', '--RK4', action="store_true", help="RK4")
    exercise.add_argument('-b', '--MC',  action="store_true", help="Monte-Carlo")
    exercise.add_argument('-c', '--VD',  action="store_true", help="Vital Dynamics")
    exercise.add_argument('-d', '--SV',  action="store_true", help="Seasonal Variation")
    exercise.add_argument('-e', '--VC',  action="store_true", help="Vaccination")
    exercise.add_argument('-f', '--Final', action="store_true", help="Combination of all models")

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args  = parser.parse_args()
    exA   = args.RK4
    exB   = args.MC
    exC   = args.VD
    exD   = args.SV
    exE   = args.VC
    exALL = args.Final

    print(parser.description)

    n = int(1e4) #nr of points for RK4 simulation

    if exA:

        print('\nExercise A: Runge-Kutta 4 \n')

        print(f'{"S":>12s}{"I":>10s}{"R":>10s}{"Sum":>10s}','\n'+'='*42)

        # Make RK4 simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R, time, f  = Solver.RK4(a, b_list[i], c, S_0, I_0, R_0, N, T, n, Basic=True, fx=Solver.fS, fy=Solver.fI)
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], title_method="Runge-Kutta 4", method='RK4', save_plot=False, folder='5a')

            # Printing the last values (at equilibrium)
            # Summing up the values to check that population is constant
            SUM = S[-1]+I[-1]+R[-1]
            print(f"{pop[i]:>2s}{S[-1]:10.2f}{I[-1]:10.2f}{R[-1]:10.2f}{SUM:10.2f}")

    if exB:

        print('\nExercise B: Monte Carlo')

        # Lists for mean/std table
        S_ABCD = []
        I_ABCD = []
        R_ABCD = []


        # Make MC simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R, time, f = Solver.MC(a, b_list[i], c, S_0, I_0, R_0, N, T)
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], title_method="Monte Carlo", method='MC', save_plot=False, folder='5b')

            # Finding the equilibrium, decided for each 
            # population by looking at different plots.

            if pop[i] == 'A':
                equ = int((len(S)/12)*8)
            if pop[i] == 'B':
                equ = int((len(S)/12)*9)
            if pop[i] == 'C':
                equ = int((len(S)/12)*10)
            else:
                equ = int((len(S)/12)*10)

            # Making a Pandas dataframe with mean and std values for all
            # populations after equilibrium

            S_ABCD.append('%.2f +/- %.2f' %(np.mean(S[equ:]), np.std(S[equ:])))
            I_ABCD.append('%.2f +/- %.2f' %(np.mean(I[equ:]), np.std(I[equ:])))
            R_ABCD.append('%.2f +/- %.2f' %(np.mean(R[equ:]), np.std(R[equ:])))

        dataframe = DataFrameSolution(S_ABCD, I_ABCD, R_ABCD)

        #exporting the dataframe as png
        dfi.export(dataframe,"Results/5b/mytable.png")

    if exC:

        print('\nExercise C: Vital dynamics')


        # Make RK4 simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R, time, f  = Solver.RK4(a, b_list[i], c, S_0, I_0, R_0, N, T, n, fx=Solver.fS, fy=Solver.fI, fz=Solver.fR, Vital=True)
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], title_method="RK4 with vital dynamics", method='RK4_vitality', save_plot=True, folder='5c', tot_pop=True)


        # Make MC simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R, time, f = Solver.MC(a, b_list[i], c, S_0, I_0, R_0, N, T, vitality=True)
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], title_method="MC with vital dynamics", method='MC_vitality', save_plot=True, folder='5c', tot_pop=True)

    if exD:

        print('\nExercise D: Seasonal variation')

        # Make RK4 simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R, time, f  = Solver.RK4(a, b_list[i], c, S_0, I_0, R_0, N, T, n, fx=Solver.fS, fy=Solver.fI, fz=Solver.fR, Season=True)
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], title_method="RK4 with seasonal variation", method='RK4_seasonal', save_plot=False, folder='5d')

        # Make MC simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R, time, f = Solver.MC(a, b_list[i], c, S_0, I_0, R_0, N, T, seasonal=True)
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], title_method="MC with seasonal variation", method='MC_seasonal', save_plot=False, folder='5d')


    if exE:

        print('\nExercise E: Vaccination')

        # Make RK4 simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R, time, f  = Solver.RK4(a, b_list[i], c, S_0, I_0, R_0, N, T, n, fx=Solver.fS, fy=Solver.fI, fz=Solver.fR, Vaccine=True)
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], title_method="RK4 with vaccine", method='RK4_vaccine', save_plot=False, folder='5e', exE=True, f=f)


        # Make MC simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R, time, f = Solver.MC(a, b_list[i], c, S_0, I_0, R_0, N, T, vaccine=True)
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], title_method="MC with vaccine", method='MC_vaccine', save_plot=False, folder='5e', exE=True, f=f)

    if exALL:

        print('\nThe SIRS model with VD, SV and vaccination')

        # Make RK4 simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R, time, f  = Solver.RK4(a, b_list[i], c, S_0, I_0, R_0, N, T, n, fx=Solver.fS, fy=Solver.fI, fz=Solver.fR, CombinedModel=True)
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], title_method="RK4 Combined model", method='RK4_combined', save_plot=True, folder='CombinedModel', tot_pop=True, exE=True, f=f)


        # Make MC simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R, time, f = Solver.MC(a, b_list[i], c, S_0, I_0, R_0, N, T, vitality=True, seasonal=True, vaccine=True)
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], title_method="MC Combined model", method='MC_combined', save_plot=True, folder='CombinedModel', tot_pop=True, exE=True, f=f)
