import sys, time, argparse

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from numba        import jit
from scipy.sparse import diags


def initialize(N):
    """
    Function that initialize v, f and u
    """

    #Initialize v
    v = np.zeros(N+2)   #include start/end points
    v[0] = 0
    v[-1] = 0

    #Initialize f
    x = np.linspace(0,1, N+2)
    h = 1/(N+1)  #step size
    f = h**2*100*np.exp(-10*x)

    #Initalize u (analytical solution)
    u = 1 - (1 - np.exp(-10))*x - np.exp(-10*x)

    return u, v, f, x

def make_A(a_i, b_i, c_i, n_i):
    """
    Generates a diagonal matrix. Takes four command line arguments.
    a_i: beneath diagonal
    b_i: diagonal
    c_i: above diagonal
    N_i: size of NxN matrix
    """
    a = a_i; b = b_i; c = c_i; N = n_i

    a_diag = a*np.ones(N-1)
    b_diag = b*np.ones(N)
    c_diag = c*np.ones(N-1)

    a_diag[0] = 0
    c_diag[-1] = 0

    return a_diag, b_diag, c_diag, N


def forward_thomas(a, b, c, f, N):
    """
    Gauss elimination (thomas): forward substitution.
    """
    bb = np.zeros_like(b)  #b tilde
    ff = np.zeros_like(f)  #f tlide

    bb[0] = b[0]
    ff[0] = f[0]

    for i in range(1, N-1):
        bb[i] = b[i] - (a[i]*c[i-1])/bb[i-1]
        ff[i] = f[i] - (a[i]*ff[i-1])/(bb[i-1])

    return bb, ff


def backward_thomas(v, ff, c, bb, N):
    """
    Gauss elimination (thomas): backward substitution.
    """
    i = N-1
    while i>=2:
        v[i-1] = (ff[i-1] - c[i-1]*v[i])/bb[i-1]
        i -= 1
    return v


def forward_special(d, f):
    """
    Gauss elimination (special): backward substitution.
    """
    dd = np.zeros_like(d)
    ff = np.zeros_like(f)

    for i in range(1, N-1):
        dd[i] = (i+1)/i
        ff[i] = f[i] + ((i-1)*ff[i-1])/i

    return dd, ff


def backward_special(v, ff):
    """
    Gauss elimination (special): backward substitution.
    """
    i = N-1
    while i>=2:
        v[i-1] = (i-1)/i*(ff[i-1] + v[i])
        i -= 1

    return v


def Gauss(v, a, b, c, f, N):
    """
    Gauss elimination (thomas)
    """
    bb, ff = forward_thomas(a,b,c,f, N)
    bb[0] = b[0]
    ff[0] = f[0]
    v = backward_thomas(v, ff, c, bb, N)
    return v


def special(v):
    """
    Gauss elimination (special)
    """
    d = b
    dd, ff = forward_special(d, f)
    dd[0] = 2
    ff[0] = f[0]
    v = backward_special(v, ff)
    return v


def plot(u, v, x, solver_name=""):
    """
    Function that plots the numerical and closed solution
    """

    plt.plot(x, v, label='v(x), numerical')
    plt.plot(x, u, label='u(x), closed solution')
    plt.xlabel("x", fontsize=16)
    plt.title("Gaussian elimination: %s, N = %g" % (solver_name, N), fontsize=14)
    plt.legend()
    plt.show()


def relative_error(N_values, epsilon):
    """
    Function that calculates the relative error
    """
    for j in range(len(N_values)):
        a, b, c, N = make_A(-1, 2, -1, int(N_values[j]))
        u, v, f, x = initialize(int(N_values[j]))
        v = Gauss(v, a, b, c, f, N)
        epsilon[j] = np.max(np.log(np.abs((v[1:-2]-u[1:-2])/u[1:-2])))

    # Plotting the relative error vs. N
    plt.title('The relative error')
    plt.plot(N_values, epsilon, 'o-')
    plt.xlabel('N')
    plt.ylabel(r'$\epsilon$')
    plt.show()

    return epsilon


if __name__ == "__main__":

    ### Running program: examples for the thomas solver ###
    # python oppgave_1b_sketch_ANNA.py -t           | Runs with default values
    # python oppgave_1b_sketch_ANNA.py -t -a 4 -b 5 | Runs with a=4 and b=5, c and n default 
    # python oppgave_1b_sketch_ANNA.py -t -n 10 -E  | Runs with n=10 and calculates relative errors

    parser = argparse.ArgumentParser(description='Project 1 in FYS4150 - Computational Physics')

    # Creating mutually exclusive group (only 1 of the arguments allowed)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-t', '--thomas',    action="store_true", help="Thomas solver")
    group.add_argument('-s', '--special',   action="store_true", help="Special solver")
    group.add_argument('-LU', '--LU_decomposition', action="store_true", help="LU decomposition (scipy)")

    # Optional arguments, default values are set
    parser.add_argument('-a', type=int, nargs='?', default= -1,  help = "value beneath the diagonal")
    parser.add_argument('-b', type=int, nargs='?', default= 2,   help = "value for the diagonal")
    parser.add_argument('-c', type=int, nargs='?', default= -1,  help = "value above the diagonal")
    parser.add_argument('-n', type=int, nargs='?', default= 100, help = "value for the NxN matrix")

    # Optional argument for calculating the relative error
    parser.add_argument('-E', action='store_true',  help = "if provided, the relative error is calculated")

    # If not provided a mutual exclusive argument, print help message
    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args  = parser.parse_args()

    T   = args.thomas
    S   = args.special
    LU  = args.LU_decomposition

    err = args.E; a_i = args.a; b_i = args.b; c_i = args.c; n_i = args.n

    print('\n');print(44*'#')
    print(parser.description)
    print(44*'#');print('\n')

    print('Using values: a=%g, b=%g, c=%g, n=%g \n' % (a_i, b_i, c_i, n_i))


    #Matrix A and diagonal vectors
    a, b, c, N = make_A(a_i, b_i, c_i, n_i)

    u, v, f, x = initialize(N)

    # Values for the relative error exercise
    N_values = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    epsilon  = np.zeros(len(N_values))


    if T:
        print(13*'-'); print('Thomas Solver');print(13*'-');print('')

        start    = time.time()
        v        = Gauss(v, a, b, c, f, N)
        end      = time.time()
        run_time = end-start

        print('Time of execution: %.10f s' %run_time)
        plot(u, v, x, solver_name='Thomas')

        if err:
            print('')
            print(30*'-');print('Calculating the relative error');print(30*'-')
            print('')
            error    = relative_error(N_values, epsilon)
            table    = {'N':N_values,'error':error}
            df       = pd.DataFrame(table, columns=['N','error'])
            print(df.to_string(index=False)); print('')
            print('The relative errors:\n', error)  # unødvendig å printe igjen?

    elif S:
        print(14*'-'); print('Special Solver');print(14*'-');print('')

        start    = time.time()
        v        = special(v)
        end      = time.time()
        run_time = end-start

        print("Time of execution: %.10f" %run_time)
        plot(u, v, x, solver_name='Special')

        if err:
            print('')
            print(30*'-');print('Calculating the relative error');print(30*'-')
            print('')
            error    = relative_error(N_values, epsilon)
            table    = {'N':N_values,'error':error}
            df       = pd.DataFrame(table, columns=['N','error'])
            print(df.to_string(index=False)); print('')
            print('The relative errors:', error)

    #elif LU:
    #    print('LU solver')