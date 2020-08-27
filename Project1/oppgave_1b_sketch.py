import numpy as np
import sys
from scipy.sparse import diags

#sketch to generate A

#command line arguments
a = float(sys.argv[1])  #below diagonal
b = float(sys.argv[2])  #diagonal
c = float(sys.argv[3])  #beneath diagonal
N = int(sys.argv[4])    #size of the NxN matrix

a_diag = a*np.ones(N-1)
b_diag = b*np.ones(N)
c_diag = c*np.ones(N-1)

diagonals = [a_diag, b_diag, c_diag]
A = diags(diagonals, [-1,0,1], shape=(N,N)).toarray() #what does [-1,0,1] do???

print(A)
