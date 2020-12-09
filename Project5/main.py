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


# Hvor skal konstanter staa igjen..?

a       = 4         # Rate of transmission
c       = 0.5       # Rate of immunity loss
bA      = 1         # Rate of recovery for population A
bB      = 2         # Rate of recovery for population B
bC      = 3         # Rate of recovery for population C
bD      = 4         # Rate of recovery for population D
b_list  = [bA, bB, bC, bD]

T   = 30            # Days
n   = 1000
N   = 400           # Nr of individuals in population
S_0 = 300           # Initial number of susceptible
I_0 = 100           # Initial number of infected
R_0 = 0             # Initial number of recovered


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
        for i in range(len(b_list)):
            S, I, time  = RK4.RK4(a, b_list[i], c, S_0, I_0, RK4.fS, RK4.fI, N, T, n)
            R           = N - S - I
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='RK4', save_plot=True)

    if exB:

        print('\nExercise B')
        # Make RK4 simulation for 4 populations, with b=[bA, bB, bC, bD]
        pop = ['A', 'B', 'C', 'D']  #titles for plots
        for i in range(len(b_list)):
            #(T, N, a, b, c, S_0, I_0, R_0)
            #(a, b, c, S_0, I_0, R_0, N, T)
            S, I, R = MC.MC(a, b_list[i], c, S_0, I_0, R_0, N, T)
            time = np.linspace(0, T, len(S))
            #R           = N - S - I
            P.plot_SIR(time, b_list[i], S, I, R, T, pop[i], method='MC', save_plot=True)

    if exC:

        print('\nExercise C')

    if exD:

        print('\nExercise D')

    if exE:

        print('\nExercise E')
