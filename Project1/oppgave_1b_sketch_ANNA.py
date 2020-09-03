import sys, time, argparse

import numpy as np
from   scipy.sparse import diags
import matplotlib.pyplot as plt


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


def forward(a, b, c, f):
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


def backward(v, ff, c, bb):
    """
    Gauss elimination: backward substitution.
    """
    i = N-1
    while i>=2:
        v[i-1] = (ff[i-1] - c[i-1]*v[i])/bb[i-1]
        i -= 1
    return v


def plot(u, v, x):

    plt.plot(x, v, label='v(x), numerical')
    plt.plot(x, u, label='u(x), closed solution')
    plt.xlabel("x", fontsize=16)
    plt.title("Gaussian elimination, N = %g" % N, fontsize=14)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Project 1")

    #group = parser.add_mutually_exclusive_group()

    #group.add_argument('-tb', '--Task1b', action="store_true", help="Task1b")

    parser.add_argument('-a', type=int, nargs='?', default=-1, help ="the a value")
    parser.add_argument('-b',type=int, nargs='?', default=2,  help = "the b value")
    parser.add_argument('-c',type=int, nargs='?', default=-1, help = "the c value")
    parser.add_argument('-n',type=int, nargs='?', default=100, help = "the n value")


    if len(sys.argv) < 1:
        sys.argv.append('--help')

    args  = parser.parse_args()

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

    #Gauss elimination
    bb, ff = forward(a,b,c,f)
    bb[0] = b[0]
    ff[0] = f[0]
    v = backward(v, ff, c, bb)


    plot(u, v, x)