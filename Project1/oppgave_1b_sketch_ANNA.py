import sys, time, argparse

import numpy as np
from   scipy.sparse import diags
import matplotlib.pyplot as plt


def initialize(N):
    """
    initialize v, f and u
    """
    #initialize v
    v = np.zeros(N+2)   #include start/end points
    v[0] = 0
    v[-1] = 0

    #initialize f
    x = np.linspace(0,1, N+2)
    h = 1/(N+1)  #step size
    f = h**2*100*np.exp(-10*x)

    #initalize u (analyrical solution)
    u = 1 - (1 - np.exp(-10))*x - np.exp(-10*x)

    return u, v, f, x

def make_A(a_i, b_i, c_i, n_i):
    """
    Generates a diagonal matrix. Takes four command line arguments.
    a: beneath diagonal
    b: diagonal
    c: above diagonal
    N: size of NxN matrix
    """
    a = a_i
    b = b_i
    c = c_i
    N = n_i

    a_diag = a*np.ones(N-1)
    b_diag = b*np.ones(N)
    c_diag = c*np.ones(N-1)

    a_diag[0] = 0
    c_diag[-1] = 0

    return a_diag, b_diag, c_diag, N


def forward_thomas(a, b, c, f):
    """
    Gauss elimination: forward substitution.
    """
    bb = np.zeros_like(b)  #b tilde
    ff = np.zeros_like(f)  #f tlide

    bb[0] = b[0]
    ff[0] = f[0]

    for i in range(1, N-1):
        bb[i] = b[i] - (a[i]*c[i-1])/bb[i-1]
        ff[i] = f[i] - (a[i]*ff[i-1])/(bb[i-1])

    return bb, ff


def backward_thomas(v, ff, c, bb):
    """
    Gauss elimination: backward substitution.
    """
    i = N-1
    while i>=2:
        v[i-1] = (ff[i-1] - c[i-1]*v[i])/bb[i-1]
        i -= 1
    return v

def forward2(d, f):
    dd = np.zeros_like(d)
    ff = np.zeros_like(f)

    for i in range(1, N-1):
        dd[i] = (i+1)/i
        ff[i] = f[i] + ((i-1)*ff[i-1])/i

    return dd, ff


def backward2(v, ff):
    i = N-1
    while i>=2:
        v[i-1] = (i-1)/i*(ff[i-1] + v[i])
        i -= 1

    return v

def Gauss(v, a, b, c):
    #Gauss elimination
    bb, ff = forward_thomas(a,b,c,f)
    bb[0] = b[0]
    ff[0] = f[0]
    v = backward_thomas(v, ff, c, bb)
    return v

def symmetric(v):
    d = b
    dd, ff = forward2(d, f)
    dd[0] = 2
    ff[0] = f[0]
    v = backward2(v, ff)
    return v


def plot(u, v, x):

    plt.plot(x, v, label='v(x), numerical')
    plt.plot(x, u, label='u(x), closed solution')
    plt.xlabel("x", fontsize=16)
    plt.title("Gaussian elimination, N = %g" % N, fontsize=14)
    plt.legend()
    plt.show()


def relative_error(N_values, epsilon, v, a, b, c):
    #ERROR 1d)
    for j in range(len(N_values)):
        a, b, c, N = make_A(-1, 2, -1, int(N_values[j]))
        u, v, f, x = initialize(int(N_values[j]))
        v = Gauss(v, a, b, c)
        epsilon[j] = np.max(np.log(np.abs((v[1:-2]-u[1:-2])/u[1:-2])))

    plt.plot(N_values, epsilon, 'o-')
    plt.xlabel("N")
    plt.show()

    print(epsilon)


if __name__ == "__main__":

    ### Running program: example for the thomas solver ###
    # python oppgave_1b_sketch_ANNA.py -t           | Runs with default values
    # python oppgave_1b_sketch_ANNA.py -t -a 4 -b 5 | Runs with a=4 and b=5, c and n default 

    parser = argparse.ArgumentParser(description="Project 1")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-t', '--thomas',    action="store_true", help="Thomas solver")
    group.add_argument('-s', '--symmetric', action="store_true", help="symmetric solver")

    parser.add_argument('-a', type=int, nargs='?', default= -1,  help = "value beneath the diagonal")
    parser.add_argument('-b', type=int, nargs='?', default= 2,   help = "value for the diagonal")
    parser.add_argument('-c', type=int, nargs='?', default= -1,  help = "value above the diagonal")
    parser.add_argument('-n', type=int, nargs='?', default= 100, help = "value for the NxN matrix")

    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    args  = parser.parse_args()

    T   = args.thomas
    S   = args.symmetric
    a_i = args.a
    b_i = args.b
    c_i = args.c
    n_i = args.n

    print(a_i)
    print(b_i)
    print(c_i)
    print(n_i)

    #Matrix A and diagonal vectors
    a, b, c, N = make_A(a_i, b_i, c_i, n_i)

    u, v, f, x = initialize(N)

    N_values = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    epsilon = np.zeros(len(N_values))


    if T:
        start = time.time()
        v     = Gauss(v, a, b, c)
        end   = time.time()
        print("Time of execution: ", end-start)
        plot(u, v, x)
        error = relative_error(N_values, epsilon, v, a, b, c)

    if S:
        # Må fikse plot så tittel osv blir riktig til hver oppgave
        start = time.time()
        v     = symmetric(v)
        end   = time.time()
        print("Time of execution: ", end-start)
        plot(u, v, x)
        error = relative_error(N_values, epsilon, v, a, b, c)