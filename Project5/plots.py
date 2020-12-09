
import matplotlib.pyplot as plt

def plot_SIR(time, b, S, I, R, T, method, save_plot=False):

    plt.figure()
    plt.plot(time, S, label="Susceptible")
    plt.plot(time, I, label="Infected")
    plt.plot(time, R, label="Recovered")
    plt.legend(fontsize=15)
    plt.title('SIRS ??? b=%g' %b)
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Nr. of individuals", fontsize=15)
    plt.xticks(fontsize=13);plt.yticks(fontsize=13)
    plt.tight_layout()

    if save_plot:
        print('\nSaving plot for method: %s, T=%g, b=%g' %(method, T, b))
        plt.savefig('Results/SIRS_%s_T[%g]_b[%g]'% (method, T, b))
    else:
        plt.show()
