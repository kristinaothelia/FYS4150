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





# Global parameters
# ??

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Solar system")
    group  = parser.add_mutually_exclusive_group()

    group.add_argument('-1', '--c4', action="store_true", help="Project 4, c)")
    group.add_argument('-2', '--d4',  action="store_true", help="Project 4, d)")

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args  = parser.parse_args()
    ex_4c = args.c4
    ex_4d = args.d4


    if ex_4c == True:
        print("--------------------------------------------------------------")
        print("2x2 latice")
        print("--------------------------------------------------------------")

        # Initial conditions
        max_cycles = 1e7            # Max. MC cycles
        max_cycles = 10000000
        L          = 2              # Number of spins
        temp       = 1              # [kT/J] Temperature
        J          = 1

        # MC
        I2.twoXtwo(L, temp, max_cycles)
        # Analytic solutions
        I2.Analythical_2x2(J, L, temp)


    elif ex_4d == True:
        print("--------------------------------------------------------------")
        print("20x20 latice")
        print("--------------------------------------------------------------")

        L = 20      # Number of spins

        T = 1.0     # [kT/J] Temperature
        T = 2.4     # [kT/J] Temperature
