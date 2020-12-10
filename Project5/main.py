"""
FYS4150 - Computational Physics

Project 5 - Disease Modelling
"""

import sys, os, time, argparse

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

import RK4_disease       as RK4
import plots             as P
import anna as MC  #change name

import dataframe_image as dfi



# Hvor skal konstanter staa igjen..?

a       = 4         # Rate of transmission
c       = 0.5       # Rate of immunity loss
bA      = 1         # Rate of recovery for population A
bB      = 2         # Rate of recovery for population B
bC      = 3         # Rate of recovery for population C
bD      = 4         # Rate of recovery for population D
b_list  = [bA, bB, bC, bD]

T   = 12          # Days
N   = 400           # Nr of individuals in population
S_0 = 300           # Initial number of susceptible
I_0 = 100           # Initial number of infected
R_0 = 0             # Initial number of recovered

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
                                                **{'color': 'blue',\
                                                   'border-color': 'white',
                                                   'font-size' : '13pt'})\
                               .set_properties(subset = pd.IndexSlice[['I'], :],\
                                               **{'color': 'red',\
                                                  'border-color': 'white',
                                                  'font-size' : '13pt'})\
                               .set_properties(subset = pd.IndexSlice[['R'], :],\
                                               **{'color': 'green',\
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

        print('\nExercise A')

        # Make RK4 simulation for 4 populations, with b=[bA, bB, bC, bD]
        pop = ['A', 'B', 'C', 'D']  #titles for plots
        n   = int(1e4) #nr of points for RK4 run
        for i in range(len(b_list)):
            S, I, _, time  = RK4.RK4(a, b_list[i], c, S_0, I_0, R_0, N, T, n, fx=RK4.fS, fy=RK4.fI)
            R  = N - S - I
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='RK4', save_plot=True)

    if exB:

        # For mean/std table
        S_ABCD = []
        I_ABCD = []
        R_ABCD = []

        print('\nExercise B')
        # Make MC simulation for 4 populations, with b=[bA, bB, bC, bD]
        pop = ['A', 'B', 'C', 'D']  #titles for plots
        for i in range(len(b_list)):
            S, I, R = MC.MC(a, b_list[i], c, S_0, I_0, R_0, N, T)
            time = np.linspace(0, T, len(S))
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='MC', save_plot=True)

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
        dfi.export(dataframe,"mytable.png")

    if exC:

        print('\nExercise C')

        # Make RK4 simulation for 4 populations, with b=[bA, bB, bC, bD]
        pop = ['A', 'B', 'C', 'D']  #titles for plots
        n   = int(1e4) #nr of points for RK4 run
        for i in range(len(b_list)):
            S, I, R, time  = RK4.RK4(a, b_list[i], c, S_0, I_0, R_0, N, T, n, fx=RK4.fS, fy=RK4.fI, fz=RK4.fR, Vital=True)
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='RK4_vitality', save_plot=True, tot_pop=True)

        # Make MC simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R = MC.MC(a, b_list[i], c, S_0, I_0, R_0, N, T, vitality=True)
            time = np.linspace(0, T, len(S))
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='MC_vitality', save_plot=True, tot_pop=True)


    if exD:

        print('\nExercise D')


        # Make RK4 simulation for 4 populations, with b=[bA, bB, bC, bD]
        pop = ['A', 'B', 'C', 'D']  #titles for plots
        n   = int(1e4) #nr of points for RK4 run
        for i in range(len(b_list)):
            S, I, R, time  = RK4.RK4(a, b_list[i], c, S_0, I_0, R_0, N, T, n, fx=RK4.fS, fy=RK4.fI, fz=RK4.fR, Vital=True, seasonal=True)
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='RK4_vitality_seasonal', save_plot=True, tot_pop=True)


        # Make MC simulation for 4 populations, with b=[bA, bB, bC, bD]
        for i in range(len(b_list)):
            S, I, R = MC.MC(a, b_list[i], c, S_0, I_0, R_0, N, T, vitality=True, seasonal=True)
            time = np.linspace(0, T, len(S))
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='MC_vitality_season', save_plot=True, tot_pop=True)

    if exE:

        print('\nExercise E')
