import numpy as np
import sys
from scipy.sparse import diags
import matplotlib.pyplot as plt


def make_A():
    #command line arguments
    a = float(sys.argv[1])  #beneath diagonal
    b = float(sys.argv[2])  #diagonal
    c = float(sys.argv[3])  #above diagonal
    N = int(sys.argv[4])    #size of the NxN matrix

    a_diag = a*np.ones(N-1)
    b_diag = b*np.ones(N)
    c_diag = c*np.ones(N-1)

    diagonals = [a_diag, b_diag, c_diag]
    A = diags(diagonals, [-1,0,1], shape=(N,N)).toarray() #what does [-1,0,1] do???

    return A, a_diag, b_diag, c_diag, N


def forward(a, b, c, f):
    bb = np.zeros_like(b)  #b tilde
    ff = np.zeros_like(f)  #f tlide

    bb[0] = b[0]
    ff[0] = f[0]

    for i in range(1, N-1):
        bb[i] = b[i] - (a[i]*c[i-1])/bb[i-1]
        ff[i] = f[i] - (a[i]*ff[i-1])/(bb[i-1])

    return bb, ff


def backward(v, ff, c, bb):
    i = N-1
    print(N)
    while i>=2:
        #print(i)
        v[i-1] = (ff[i-1] - c[i-1]*v[i])/bb[i-1]
        i -= 1
    return v

def plaot(u, v, x):
    return None

#matrix A and diagonal vectors
A, a, b, c, N = make_A()
a[0] = 0
c[-1] = 0

#print(A)
#print(N)


#initialize v
v = np.zeros(N+2)   #include start/end points
v[0] = 0
v[-1] = 0
#print(v)


#initialize f
x = np.linspace(0,1, N+2)
#print(x)
h = 1/(N+1)
f_function = h**2*100*np.exp(-10*x)


bb, ff = forward(a,b,c,f_function)
bb[0] = b[0]
ff[0] = f_function[0]
v = backward(v, ff, c, bb)

v_closed = 1 - (1 - np.exp(-10))*x - np.exp(-10*x)

#print(v[0])


plt.plot(x, v, label='numerical')
plt.plot(x, v_closed, label='closed solution')
plt.legend()
plt.show()
