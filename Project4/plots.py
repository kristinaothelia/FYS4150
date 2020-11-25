"""
Various plotting functions used in main.py
"""

import sys

import matplotlib.pyplot as plt
import numpy             as np


def plot_MCcycles_vs_err(mc_cycles, error):
    """Plotting error vs. number of MC cycles.

    loglog or semilog?
    Need better adjustment of plot.
    New title, xlabel, ylabel etc.
    """
    plt.figure(figsize=(15, 10))

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
                     rotation='0',   # plot seems weird w/angle other than 0 or 360
                     va='top',       #  [ 'center' | 'top' | 'bottom' | 'baseline' ]
                     ha='right')     #  [ 'left' | 'right' | 'center']


    plt.title('Error of the Mean Abs. Magnetization',fontsize=15)
    plt.xlabel('Number of Monte-Carlo Cycles',fontsize=15)
    plt.ylabel('error',fontsize=15)
    plt.xticks(fontsize=13);plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.savefig(f'results/plots/4c/ErrorMeanMagnetizationAbs')
    plt.show()


def expected_vals_two_temp(MCcycles, T1, T2, expected):
    """
    Function for plotting expectation values vs. MC cycles,
    - two temperatures in two different initial states.

    expected has structure [temp, physical quantity, ...]
    """

    names     = ['Energy','Magnetization','Abs. Magnetization',\
                 'Specific Heat','Susceptibility']

    ylabels   = [r'$\langle E\rangle$', r'$\langle M \rangle$',\
                 r'$\langle|M|\rangle$', r'$C_V$', r'$\chi$']

    save_as   = ['energy','mag','Mabs','CV','CHI']

    x = np.linspace(1,MCcycles,MCcycles, endpoint=True).astype(np.float_)

    for i, val in enumerate(expected):

        print("\nSaving plot for: ", names[i])

        plt.figure(figsize=(7,5))
        plt.semilogx(x, val[0,0,:], linewidth=1.0,\
                                    label=f'T={T1} (order)',\
                                    color='tab:blue')
        plt.semilogx(x, val[0,1,:], linewidth=1.0,\
                                    label=f'T={T2} (order)',\
                                    color='tab:red')

        plt.semilogx(x, val[1,0,:], '--', linewidth=1.0,\
                                          label=f'T={T1} (disorder)',\
                                          color='tab:blue')
        plt.semilogx(x, val[1,1,:], '--', linewidth=1.0,\
                                          label=f'T={T2} (disorder)',\
                                          color='tab:red')

        plt.title(f'Expectation Values of {names[i]}', fontsize=15)
        plt.xlabel('Number of Monte-Carlo Cycles', fontsize=15)
        plt.ylabel(f'{ylabels[i]}', fontsize=15)
        plt.xticks(fontsize=13);plt.yticks(fontsize=13)
        plt.legend(fontsize=13)
        plt.tight_layout()
        plt.savefig(f'results/plots/4d/expected_{save_as[i]}_{MCcycles:.1e}.png')
        plt.show()

def plot_n_accepted(MCcycles, Naccs, T1, T2):

    x  = np.linspace(1,MCcycles,MCcycles, endpoint=True).astype(np.float_)

    plt.figure(figsize=(7,5))

    plt.plot(x, Naccs[0,0,:],label=f'T={T1} (order)',color='tab:blue')
    plt.plot(x, Naccs[0,1,:],label=f'T={T2} (order)',color='tab:red')
    plt.plot(x, Naccs[1,0,:], '--',label=f'T={T1} (disorder)',color='tab:blue')
    plt.plot(x, Naccs[1,1,:], '--',label=f'T={T2} (disorder)',color='tab:red')

    plt.title('Total Number of Accepted Configurations', fontsize=15)
    plt.xlabel('Number of Monte-Carlo Cycles', fontsize=15)
    plt.ylabel('Number of Accepted Configurations', fontsize=15)
    plt.xticks(fontsize=13);plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(f'results/plots/4d/AcceptedConfigs')
    plt.show()

def plot_4f(name, ylabel, save_as):
    plt.title(f'Phase transition for {name}', fontsize=15)
    plt.xlabel('Temperature [kT/J]', fontsize=15)
    plt.ylabel(f'{ylabel}', fontsize=15)
    plt.xticks(fontsize=13);plt.yticks(fontsize=13)
    plt.legend(fontsize=13)
    plt.tight_layout()
    #plt.savefig(f'results/plots/4f/PT_{save_as}.png')
