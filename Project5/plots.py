import matplotlib.pyplot as plt

"""
Colors
black               | '#000000'
dimgray             | '#696969'

Green:
green               | '#008000'
limegreen           | '#32CD32'
forestgreen         | '#228B22'


Red:
red                 | '#FF0000'
darkred             | '#8B0000'
orangered           | '#FF4500'
firebrick           | '#B22222'

Blue/Purple
midnightblue        | '#191970'
mediumblue          | '#0000CD'
royalblue           | '#4169E1'
purple              | '#800080'
indigo              | '#4b0082'
darkslateblue       | '#483D8B'
"""

def plot_SIR(time, b, S, I, R, T, pop, method, save_plot=False, folder='', tot_pop=False, exE=False):
    """
    tot_pop     | Total population line
    exE         | Add vertical line to indicate start of vaccination
    """
    plt.figure()
    plt.plot(time, S, label="Susceptible", color='#4169E1') # Blue/Purple
    plt.plot(time, I, label="Infected", color='#B22222')    # Red
    plt.plot(time, R, label="Recovered", color='#228B22')   # Green

    if tot_pop:
        plt.plot(time, (S+I+R), label="Tot. population", color='#FF4500') # '#4b0082'
    if exE:
        plt.axvline(6.0, linestyle="--", label="Vaccination start", color="gray")   # T/2?

    plt.legend(fontsize=15)
    plt.title('Disease evolution in population %s \nMethod: %s. b=%g' %(pop, method, b), fontsize=15)
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Nr. of individuals", fontsize=15)
    plt.xticks(fontsize=14);plt.yticks(fontsize=14)
    plt.tight_layout()

    if save_plot:
        print('\nSaving plot for method: %s, T=%g, b=%g' %(method, T, b))
        plt.savefig('Results/%s/SIRS_%s_T[%g]_b[%g]'% (folder, method, T, b))
    else:
        plt.show()
