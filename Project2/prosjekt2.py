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

	return A


def MaxNonDiag(A, n):
	'''
	Function to find the maximum non diagonal element
	'''

	maxnondiag = 0.0

	for i in range(n):
		for j in range(i+1, n):

			if abs(A[i,j]) > maxnondiag:
				maxnondiag = abs(A[i,j])
				#l = i
				#k = j
				k = i
				l = j


	return maxnondiag, k, l


def Jacobi(A, n, epsilon=1e-8):

	max_it     = 1000  #n**3?
	iterations = 0 

	A = np.array(A)
	R = np.eye(n)

	#maxnondiag, k, l = MaxNonDiag(A,n)
	maxnondiag = 0.0
	#for i in range(n):
	#	for j in range(i+1, n):
	#		if abs(A[i,j]) > maxnondiag:
	#			maxnondiag = abs(A[i,j])


	# FEILMELDING!!!!!
	#File "prosjekt2.py", line 62, in Jacobi
    #while (maxnondiag < epsilon) and (iterations <= max_it):
	#TypeError: '<' not supported between instances of 'tuple' and 'float'

	while (maxnondiag > epsilon) and (iterations <= max_it):
		maxnondiag = 0.0
		k = 0
		l = 0
		maxnondiag = MaxNonDiag(A,n)
		A, R       = Jacobi_rotation(A, R, k, l, n)
		iterations += 1

	EigenVec = R
	EigenVal = np.diag(A)
	return EigenVal, EigenVec, iterations



def Jacobi_rotation(A, R, k, l, n):
	'''

	'''

	if A[k,l] != 0:

		tau = (A[l,l] - A[k,k]) / (2 * A[k,l])
		if tau > 0:
			t = -tau + np.sqrt(1 + tau**2)
		else:
			t = -tau - np.sqrt(1 + tau**2)

		c = 1 / np.sqrt(1 + t**2)
		s = c * t

	else:
		c = 1   # cos(theta)?
		s = 0   # sin(theta)?

	a_kk = A[k,k]
	a_ll = A[l,l]

	A[k,k] = c**2 * a_kk - 2 * c * s * A[k,l] + s**2 * a_ll
	A[l,l] = s**2 * a_kk + 2 * c * s * A[k,l] + c**2 * a_ll
	A[k,l] = 0 
	A[l,k] = 0

	for i in range(n):

		if i != k and i != l:
			a_ik   = A[i,k]
			a_il   = A[i,l]
			A[i,k] = c*a_ik - s*a_il
			A[k,i] = A[i,k]
			A[i,l] = c*a_il + s*a_ik
			A[l,i] = A[i,l]

		# new eigenvectors
		r_ik = R[i,k]
		r_il = R[i,l]

		R[i,k] = c*r_ik - s*r_il
		R[i,l] = c*r_il + s*r_ik

	return A, R




if __name__ == "__main__":

	N = 100

	rho_0 = 0   # rho min
	rho_N = 1   # rho max 

	h = (rho_N-rho_0)/N    # step length

	diag = 2/h**2
	non_diag = -1/h**2
	
	A = Toeplitz(N, diag, non_diag)
	print(A)

	# diagonalize and obtain eigenvalues, not necessarily sorted
	EigValues_np, EigVectors_np = np.linalg.eig(A)
	#print(EigValues_np, EigVectors_np)


	EigenVal, EigenVec, iterations = Jacobi(A, N, epsilon=0.0001)
	print(iterations)

	# sort eigenvectors and eigenvalues
	permute      = EigenVal.argsort()
	EigenValues  = EigenVal[permute]
	EigenVectors = EigenVec[:,permute]

	# now plot the results for the three lowest lying eigenstates
	for i in range(3):
		print(EigenValues[i])

	# For plotting 
	#x = np.linspace(1,N,N-1)
	#x = diag + 2*non_diag*np.cos((x*np.pi)/(N))