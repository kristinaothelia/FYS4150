import sys, time, argparse

import numpy             as np
import matplotlib.pyplot as plt

# should imports be under name/main?

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
				k = i
				l = j


	return maxnondiag, k, l


def Jacobi(A, n, epsilon=1e-8, max_it=1e4):

	#max_it     = n**3
	iterations = 0

	A = np.array(A)
	R = np.eye(n)

	maxnondiag, k, l = MaxNonDiag(A,n)

	while (maxnondiag > epsilon) and (iterations <= max_it):

		maxnondiag, k, l = MaxNonDiag(A,n)
		A, R             = Jacobi_rotation(A, R, k, l, n)
		iterations += 1

	EigenVec = R
	EigenVal = np.diag(A)

	return EigenVal, EigenVec, iterations


def analytical_eigenpairs(n, d, a):
	"""
	Computes the analytical values for the eigenpair for buckling beam
	problem.
	"""
	j = np.linspace(1, n)  #j = 1,2,...,(n-1)
	lam = d + 2*a*np.cos(j*np.pi/n)
	u = np.zeros([n, n])
	for j in range(1, n):
		for i in range(1, n-1):
			u[j,i] = np.sin(i*j*np.pi/n)
		u[j,:] /= np.linalg.norm(u[j,:])   #normalizing
	return lam, u



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

	parser = argparse.ArgumentParser(description='Project 2 in FYS4150 - Computational Physics')

 	# Creating mutually exclusive group (only 1 of the arguments allowed at each time)
	# Mutually exclusive arguments must be optional
	group = parser.add_mutually_exclusive_group()
	group.add_argument('-B', '--beam',    action="store_true", help="The buckling beam problem")
	group.add_argument('-Q', '--quantum', action="store_true", help="Quantum mechanics")

	# Optional arguments for input values, default values are set
	parser.add_argument('-n', type=int, nargs='?', default= 100,  help="dimensionality of matrix")


	# If not provided a mutual exclusive argument, print help message
	if len(sys.argv) <= 1:
		sys.argv.append('--help')

	args  = parser.parse_args()

	BucklingBeam       = args.beam
	QuantumMechanics   = args.quantum

	#N 	   = args.n
	N = 10
	max_it = N**3

	rho0 = 0   # rho min
	rhoN = 1   # rho max

	h = (rhoN-rho0)/N     # step length

	diag = 2/h**2
	non_diag = -1/h**2

	if BucklingBeam:

		A = Toeplitz(N, diag, non_diag)

		# diagonalize and obtain eigenvalues, not necessarily sorted
		#EigValues_np, EigVectors_np = np.linalg.eig(A)  # eigenvectors are negative
		EigValues_np, EigVectors_np = np.linalg.eig(A)

		# sort eigenvectors and eigenvalues
		permute         = EigValues_np.argsort()
		EigenValues_np  = EigValues_np[permute]
		EigenVectors_np = EigVectors_np[:,permute]
		#print(EigenValues_np)
		#print(EigenVectors_np)


		#testing with numpy
		#if v = eigenvector and lam=eigen value, then they should work like
		#Av = lam*v
		#print(A@EigVectors_np[:,0], 'test')               #Av
		#print(EigValues_np[0]*EigVectors_np[:,0]) #lam*v

		EigenVal, EigenVec, iterations = Jacobi(A, N, epsilon=1e-8, max_it=max_it)
		#print(iterations)
		#print("-------------------------")

		#testing with own method
		#print(A@EigenVec[:,0], 'test')
		#print(EigenVal[0]*EigenVec[:,0])

		# sort eigenvectors and eigenvalues
		permute      = EigenVal.argsort()
		EigenValues  = EigenVal[permute]
		EigenVectors = EigenVec[:,permute]
		#print("-------------")
		#print(EigenValues)


		FirstEigvector_np   = EigenVectors_np[:,0]
		FirstEigvector_Jac  = EigenVectors[:,0]
		SecondEigvector_np  = EigenVectors_np[:,1]
		SecondEigvector_Jac = EigenVectors[:,1]

		"""
		print('Comparing eigenvector for the lowest eigenvalue')
		print(FirstEigvector_np)
		print(FirstEigvector_Jac)
		"""

		###ANALYTICAL test
		lam_eigen, u_eigen = analytical_eigenpairs(N, diag, non_diag)  #first level 1, not 0??

		FirstEigvector_analytical = u_eigen[1,:]
		SecondEigvector_analytical = u_eigen[2,:]

		# Plotting the eigenvector for the lowest eigenvalue
		rho = np.linspace(rho_0, rho_N, len(FirstEigvector_np))
		#plt.plot(rho, FirstEigvector_np, label='numpy')
		plt.plot(rho, FirstEigvector_analytical, label='analytical 1')
		plt.plot(rho, FirstEigvector_Jac, label='Jacobi 1')
		#plt.plot(rho, SecondEigvector_np, label='numpy')
		plt.plot(rho, SecondEigvector_analytical, label='analytical 2')
		plt.plot(rho, SecondEigvector_Jac, label='Jacobi 2')
		plt.legend()
		plt.show()