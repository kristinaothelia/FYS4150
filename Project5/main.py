"""
FYS4150 - Computational Physics

Project 5 - Disease Modelling 
"""


import sys, os, time, argparse

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd



if __name__ == '__main__':

    parser    = argparse.ArgumentParser(description=__doc__)
    exercise  = parser.add_mutually_exclusive_group()

    exercise.add_argument('-a', '--RK4', action="store_true", help="RK4")
    exercise.add_argument('-b', '--MC', action="store_true", help="Monte-Carlo")
    exercise.add_argument('-c', '--VD', action="store_true", help="Vital Dynamics")
    exercise.add_argument('-d', '--SV', action="store_true", help="Seasonal Variation")
    exercise.add_argument('-e', '--VC', action="store_true", help="Vaccination")

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args = parser.parse_args()
    exA  = args.RK4
    exB  = args.MC
    exC  = args.VD
    exD  = args.SV
    exE  = args.VC 

    print(parser.description)