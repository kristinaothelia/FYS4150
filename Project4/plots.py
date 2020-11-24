"""
Various plotting functions used in main.py
"""

import matplotlib.pyplot as plt
import numpy             as np

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
                     rotation='0',   # plot seems weird w/angle other than 0 or 360..?
                     va='top',       #  [ 'center' | 'top' | 'bottom' | 'baseline' ]
                     ha='right')     #  [ 'left' | 'right' | 'center']

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

        print("Saving plot for: ", names[i])

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
        plt.savefig(f'results/plots/4d/expected_{save_as[i]}_{MCcycles}')
        #plt.show()

def plot_n_accepted(MCcycles, Naccs, T1, T2):

    #print(Naccs)
    #print(Naccs.shape)

    order_T1 = Naccs[0,0,:]

    x  = np.linspace(1,MCcycles,MCcycles, endpoint=True).astype(np.float_)
    na = np.linspace(1,MCcycles,MCcycles, endpoint=True)

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
    plt.savefig(f'results/plots/4f/PT_{save_as}.png')
