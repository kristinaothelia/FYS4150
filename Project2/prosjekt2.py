import sys, time, argparse

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt


# http://arma.sourceforge.net/docs.html#eig_sym
# http://compphysics.github.io/ComputationalPhysics/doc/pub/eigvalues/pdf/eigvalues-print.pdf


def Toeplitz(n, diag, non_diag):
	'''
	Tridiagonal Toeplitz (nxn)
	Used to compare results from the Jakobi method???
	'''

	A = diag*np.eye(n) + non_diag*np.eye(n, k = -1) + non_diag*np.eye(n, k=1)

	# diagonalize and obtain eigenvalues, not necessarily sorted
	EigValues, EigVectors = np.linalg.eig(A)  

	return A, EigValues, EigVectors


def MaxNonDiag(A):
	'''
	Function to find the maximum non diagonal element
	'''
	pass


def Jacobi(A, epsilon=1e-8):

	max_it = 1000
	iterations = 0 

	while maxnondiag > tolerance and iterations <= maxiter:
		maxnondiag = MaxNonDiag(A)
		Jacobi_rotation()
		iterations += 1

	pass



def Jacobi_rotation():
	pass




if __name__ == "__main__":

	N = 100

	rho_0 = 0   # rho min
	rho_N = 1   # rho max 

	h = (rho_N-rho_0)/N    # step length

	diag = 2/h**2
	non_diag = -1/h**2
	
	A, EigValues, EigVectors = Toeplitz(N, diag, non_diag)
