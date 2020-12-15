"""
------------------------------------------------------------------------
FYS4150 Computational Physics: Project 5
Disease Modelling using Runge-Kutta 4 and Monte Carlo (on SIRS model)
------------------------------------------------------------------------
"""
# Note: Need to pip install dataframe_image

import sys, os, time, argparse

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import RK4_MonteCarlo_v2   as Solver
import plots             as P
import dataframe_image   as dfi

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

T   = 12            # Time [??]
N   = 400           # Nr of individuals in population
S_0 = 300           # Initial number of susceptible
I_0 = 100           # Initial number of infected
R_0 = 0             # Initial number of recovered

# Functions to make mean/std table in ex. b)
def magnify():
    return [dict(selector="th",
                 props=[('text-align', 'center'), ("font-size", "13pt")]),
            #dict(selector="td",
             #    props=[('padding', "0em 0em")]),
            #dict(selector="th:hover",
             #    props=[("font-size", "12pt")]),
            #dict(selector="tr:hover td:hover",
             #    props=[('max-width', '200px'),
              #          ('font-size', '12pt')])
              ]

def DataFrameSolution(S_arr, I_arr, R_arr):
    """Function creating dataframe with solution values.

    Parameters
    ----------

    Returns
    -------
    DataFrame object
    """

    groups = ['S', 'I', 'R']

    print(S_arr)
    print(I_arr)
    print(R_arr)

    dataframe = pd.DataFrame([S_arr, I_arr, R_arr], columns=['A', 'B', 'C', 'D'], index=groups) #.transpose()

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

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args = parser.parse_args()
    exA  = args.RK4
    exB  = args.MC
    exC  = args.VD
    exD  = args.SV
    exE  = args.VC

    print(parser.description)


    if exA:

        print('\nExercise A: Runge-Kutta 4')

        # Make RK4 simulation for 4 populations, with b=[bA, bB, bC, bD]
        n   = int(1e4) #nr of points for RK4 run
        for i in range(len(b_list)):
            S, I, _, time  = Solver.RK4(a, b_list[i], c, S_0, I_0, R_0, N, T, n, Basic=True, fx=Solver.fS, fy=Solver.fI)
            R  = N - S - I
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='RK4', save_plot=True, folder='5a')

    if exB:

        print('\nExercise B: Monte Carlo')

        # For mean/std table
        S_ABCD = []
        I_ABCD = []
        R_ABCD = []

        # Make MC simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R = Solver.MC(a, b_list[i], c, S_0, I_0, R_0, N, T)
            time = np.linspace(0, T, len(S))
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='MC', save_plot=True, folder='5b')

            # Finding the equilibrium, decided for each population by looking
            # at the plots. For population A-C: T=6, for D: T=10
            if pop[i] == 'D':
                equ = int((len(S)/12)*10)
            else:
                equ = int((len(S)/12)*6)

            # Making a Pandas ddataframe with mean and std values for all
            # populations after equilibrium

            S_ABCD.append('%.2f +/- %.2f' %(np.mean(S[equ:]), np.std(S)))
            I_ABCD.append('%.2f +/- %.2f' %(np.mean(I[equ:]), np.std(I)))
            R_ABCD.append('%.2f +/- %.2f' %(np.mean(R[equ:]), np.std(R)))

        dataframe = DataFrameSolution(S_ABCD, I_ABCD, R_ABCD)

        #exporting the dataframe as png
        dfi.export(dataframe,"Results/5b/mytable.png")

    if exC:

        print('\nExercise C: Vital dynamics')


        # Make RK4 simulation for 4 populations, with b=[bA, bB, bC, bD]
        n   = int(1e4) #nr of points for RK4 run
        for i in range(len(b_list)):
            S, I, R, time  = Solver.RK4(a, b_list[i], c, S_0, I_0, R_0, N, T, n, fx=Solver.fS, fy=Solver.fI, fz=Solver.fR, Vital=True)
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='RK4_vitality', save_plot=True, folder='5c', tot_pop=True)


        # Make MC simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R = Solver.MC(a, b_list[i], c, S_0, I_0, R_0, N, T, vitality=True)
            time = np.linspace(0, T, len(S))
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='MC_vitality', save_plot=True, folder='5c', tot_pop=True)


    if exD:

        print('\nExercise D: Seasonal variation')


        # Make RK4 simulation for 4 populations, with b=[bA, bB, bC, bD]
        n   = int(1e4) #nr of points for RK4 run
        for i in range(len(b_list)):
            S, I, R, time  = Solver.RK4(a, b_list[i], c, S_0, I_0, R_0, N, T, n, fx=Solver.fS, fy=Solver.fI, fz=Solver.fR, Season=True)
            R = N - S - I
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='RK4_seasonal', save_plot=True, folder='5d')



        # Make MC simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R = Solver.MC(a, b_list[i], c, S_0, I_0, R_0, N, T, seasonal=True)
            time = np.linspace(0, T, len(S))
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='MC_seasonal', save_plot=True, folder='5d')


    if exE:

        print('\nExercise E: Vaccination')

        # Make RK4 simulation for 4 populations, with b=[bA, bB, bC, bD]
        n   = int(1e4) #nr of points for RK4 run
        for i in range(len(b_list)):
            S, I, R, time  = Solver.RK4(a, b_list[i], c, S_0, I_0, R_0, N, T, n, fx=Solver.fS, fy=Solver.fI, fz=Solver.fR, Vaccine=True)
            #R = N-S-I
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='RK4_vaccine', save_plot=True, folder='5e')


        # Make MC simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R = Solver.MC(a, b_list[i], c, S_0, I_0, R_0, N, T, vaccine=True)
            time = np.linspace(0, T, len(S))
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='MC_vaccine', save_plot=True, folder='5e')
