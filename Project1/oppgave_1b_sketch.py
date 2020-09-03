import numpy as np
import sys
from scipy.sparse import diags
import matplotlib.pyplot as plt


def make_A():
    """
    Generates a diagonal matrix. Takes four command line arguments.
    a: beneath diagonal
    b: diagonal
    c: above diagonal
    N: size of NxN matrix
    """
    a = float(sys.argv[1])
    b = float(sys.argv[2])
    c = float(sys.argv[3])
    N = int(sys.argv[4])

    a_diag = a*np.ones(N-1)
    b_diag = b*np.ones(N)
    c_diag = c*np.ones(N-1)

    a_diag[0] = 0
    c_diag[-1] = 0

    #diagonals = [a_diag, b_diag, c_diag]
    #A = diags(diagonals, [-1,0,1], shape=(N,N)).toarray() #what does [-1,0,1] do???

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
    print(N)
    while i>=2:
        #print(i)
        v[i-1] = (ff[i-1] - c[i-1]*v[i])/bb[i-1]
        i -= 1
    return v




#matrix A and diagonal vectors
a, b, c, N = make_A()

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

#Gauss elimination
bb, ff = forward(a,b,c,f)
bb[0] = b[0]
ff[0] = f[0]
v = backward(v, ff, c, bb)

#Specialized algorithm (name?)


plt.plot(x, v, label='v(x), numerical')
plt.plot(x, u, label='u(x), closed solution')
plt.xlabel("x", fontsize=16)
plt.title("Gaussian elimination, N = %g" % N, fontsize=14)
plt.legend()
plt.show()
