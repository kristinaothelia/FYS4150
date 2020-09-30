import sys, time, argparse

#from tabulate import tabulate

import numpy             as np
import pandas            as pd
import seaborn 			 as sns
import matplotlib.pyplot as plt

# http://arma.sourceforge.net/docs.html#eig_sym
# http://compphysics.github.io/ComputationalPhysics/doc/pub/eigvalues/pdf/eigvalues-print.pdf

def Toeplitz(n, diag, non_diag):
	"""
	Tridiagonal Toeplitz (nxn)
	"""
	A = diag*np.eye(n) + non_diag*np.eye(n, k = -1) + non_diag*np.eye(n, k=1)

	return A


def EigenPairsNumpy(A):
	"""
	Function that calculates eigenpairs using numpy
	"""
	start                = time.time()
	EigVal_np, EigVec_np = np.linalg.eig(A) # eigh
	end                  = time.time()
	numpy_cpu            = (end-start)
	EigVal_np, EigVec_np = SortEigenpairs(EigVal_np, EigVec_np)

	return EigVal_np, EigVec_np, numpy_cpu


def AnalyticalEigenpairs(n, d, a):
	"""
	Computes the analytical values for the eigenpairs
	for the buckling beam problem.
	"""
	# should it be n-1??
	j   = np.linspace(1, n)  #j = 1,2,...,(n-1)
	lam = d + 2*a*np.cos(j*np.pi/n)
	u   = np.zeros([n, n])

	for j in range(1, n):
		for i in range(1, n-1):
			u[j,i] = np.sin(i*j*np.pi/n)
		u[j,:] /= np.linalg.norm(u[j,:])   # normalizing

	return lam, u


def SortEigenpairs(EigVal, EigVec):
	"""
	Function that sorts eigenvectors and eigenvalues.
	"""
	permute = EigVal.argsort()
	EigVal  = EigVal[permute]
	EigVec  = EigVec[:,permute]

	return EigVal, EigVec

def create_df(analyticals, numpys, jacobis):
	"""
	Function returning a pandas dataframe
	"""
	columns = ['analytical', 'numpy', 'jacobi']
	eigvals = np.stack([analyticals, numpys, jacobis], axis=1)
	df      = pd.DataFrame(eigvals, columns=columns)

	return df.round(4) # 4 decimals


def MaxNonDiag(A, n):
	"""
	Function to find the maximum non diagonal element
	"""
	maxnondiag = 0.0

	for i in range(n):
		for j in range(i+1, n):
			### >
			if abs(A[i,j]) >= maxnondiag:
				maxnondiag = abs(A[i,j])
				k = i
				l = j

	return maxnondiag, k, l


def Jacobi(A, n, epsilon=1e-8, max_it=1e4):
	"""
	The Jacobi method for finding eigenpairs
	"""
	start      = time.time()
	iterations = 0

	A = np.array(A)
	R = np.eye(n)

	maxnondiag, k, l = MaxNonDiag(A,n)

	while (maxnondiag > epsilon) and (iterations <= max_it):

		A, R             = JacobiRotation(A, R, k, l, n)
		maxnondiag, k, l = MaxNonDiag(A,n)
		iterations 		+= 1

	# ha med noe sånt????
	#if maxnondiag >= epsilon:
	#	print('Non diagonals are not zero, increase max_it')
	#	sys.exit()

	EigenVec = R
	EigenVal = np.diag(A)   # extracts diagonal
	end      = time.time()
	cpu_time = end-start

	return EigenVal, EigenVec, iterations, cpu_time



def JacobiRotation(A, R, k, l, n):
	"""
	Jacobi rotation
	"""

	if A[k,l] != 0:

		tau = (A[l,l] - A[k,k]) / (2 * A[k,l])
		# rewrite tau-expression to avoid numerical issues
		if tau >= 0:
			#t = -tau + np.sqrt(1 + tau**2)
			t = 1.0/(tau + np.sqrt(1.0 + tau**2))
		else:
			#t = -tau - np.sqrt(1 + tau**2)
			t = -1.0/(-tau + np.sqrt(1.0 + tau**2))

		c = 1 / np.sqrt(1 + t**2)
		s = c * t

	else:
		c = 1   # cos(theta)?
		s = 0   # sin(theta)?

	a_kk = A[k,k] # diagonalen
	a_ll = A[l,l] # diagonalen

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


def potential(r, electron=1, w=None):
	"""
	The harmonic oscillator potential
	"""
	if electron == 1:
		return r*r
	elif electron == 2:
		return w**2 * r**2 + 1/r

def rho(N, rho0, rhoN, h, electron=1, w=0.01):
	"""
	Calculate array of potential values
	"""
	v = np.zeros(N)
	r = np.linspace(rho0,rhoN,N)
	lOrbital = 0
	OrbitalFactor = lOrbital*(lOrbital+1.0)

	if electron == 1:
		# 2d) 3D, 1 electron
		for i in range(N):
			r[i] = rho0 + (i+1) * h
			v[i] = potential(r[i], electron)
	elif electron == 2:
		# 2e) 3D, 2 electron
		for i in range(N):
			r[i] = rho0 + (i+1) * h
			v[i] = potential(r[i], electron, w)
	return v

def plot_eigenvectors(rho0, rhoN, N, eigenvectors, labels=None, save=False, BB = True):
    """
    Justere figsize
    BB: Boolean statement to assign labels to x and y axes. If True, it will plot
    the vetical displacement of the buckling beam from its equilibrium point.
    If False it will plot the probability distribution of the position of the
    electrons inside the harmonic oscillator potential.
    """
    #y_ = np.max(eigenvectors)
    r_ = np.linspace(rho0, rhoN, N)
    fig, ax = plt.subplots(figsize=(7,5))

    for i in range(len(eigenvectors)):
    	plt.plot(r_, eigenvectors[i], label=labels[i])

    #plt.axis([0.0,rhoN,0.0, y_])
    plt.legend()
    # Er xlabel og ylabel riktig? Både for beam og quantum??
    if BB:
    	plt.xlabel(r'$\rho$', fontsize=15)
    	plt.ylabel(r'$u(\rho)$', fontsize=15)
    	plt.title('Eigenvectors - The Buckling Beam', fontsize=15)
    	if save == True:
    		plt.savefig('Results/EigVec_r0[%g]_rN[%g]_N[%g].png' % (rho0, rhoN, N))
    else:
    	plt.xlabel(r'$\rho$', fontsize=15)
    	plt.ylabel(r'$|u(\rho)|^2$', fontsize=15)
    	plt.title(r'Eigenvectors for varying $\omega$', fontsize=15)
    	if save == True:
    		plt.savefig('Results/LowestEigVec_omega.png')
    plt.show()

def N_iterations(N_list, diag, non_diag):
	"""
	iterations, cpu times
	"""
	it_list    = []
	cpu_jacobi = []
	cpu_numpy  = []

	for i in range(len(N_list)):

		N = N_list[i]
		A = Toeplitz(N, diag, non_diag)

		# Calculate analytical eigenpairs
		lam_eigen, u_eigen = AnalyticalEigenpairs(N, diag, non_diag)  # first level 1, not 0

		# Calculate eigenpairs with Jacobi and sort
		EigVal, EigVec, iterations, cpu = Jacobi(A, N, epsilon=1e-8, max_it=10**6)
		EigVal_Jac, EigVec_Jac          = SortEigenpairs(EigVal, EigVec)
		it_list.append(iterations)

		# CPU times
		EigVal_np, EigVec_np, numpy_cpu = EigenPairsNumpy(A)
		cpu_numpy.append(numpy_cpu)
		cpu_jacobi.append(cpu)

	# Troooor dette er riktig? For faa desimaler..?
	columns   = ['numpy', 'jacobi']
	cpu_times = np.stack([cpu_numpy, cpu_jacobi], axis=1)
	df        = pd.DataFrame(cpu_times, columns=columns).round(2)
	df.to_csv('Results/cpu_times.txt', index=None, sep='\t', mode='a')

	# Er dette riktig????
	fit = np.polyfit(N_list, it_list, 2)
	print(fit)
	plt.plot(N_list, it_list, label=(r'%.2f $N^2$' %fit[0]))
	plt.xlabel('Size of a NxN matrix', fontsize=15)
	plt.ylabel('Number of iterations', fontsize=15)
	plt.title('Similarity transformations', fontsize=15)
	plt.legend()
	plt.savefig('Results/BucklingBeam_N_iterations.png')
	plt.show()

def real_lambdas(n):
    return (4*n + 3)

def N_rho(N, rho0, rhoN_list, h, diag, non_diag):
    """
    Siden vi tidligere har sett at Jacobi
    gir veldig likt svar som numpy (men bruker mye lenger tid)
    så er det kanskje greit å bruke bare numpy for å finne rho?
    Kan jo kanskje ikke anta Jacobi er like bra for alle N, men...
    ja, ihvertfall mens vi prøver å få ting til å funke :)
    """

    # Analytical eigenvalues
    lam_eigen = [3, 7, 11, 15]
    err       = np.zeros((len(rhoN_list),len(N_list)))

    for i in range(len(rhoN_list)):
        for j in range(len(N_list)):

            h_new = (rhoN_list[i] - rho0)/N_list[j]
            diag     = 2/h**2   		# diagonal elements
            non_diag = -1/h**2  		# non-diagonal elements

            rho_values = rho(N_list[j], rho0, rhoN_list[i], h_new, electron=1)
            new_diag   = diag+rho_values
            A 		   = Toeplitz(N_list[j], new_diag, non_diag)

            # Calculate eigenpairs with numpy and sort
            EigVal_np, EigVec_np = np.linalg.eig(A) # eigh
            EigVal_np, EigVec_np = SortEigenpairs(EigVal_np, EigVec_np)
            #print(EigVal_np)
            #sys.exit(1)

            #print(EigVal_np[0])
            #err[i,j] = np.max(abs(EigVal_np-real_lambdas(np.arange(N_list[j])))) # or mean????
            #err[i,j] = abs(EigVal_np[0] - lam_eigen[0])
            err[i,j] = np.mean(abs(EigVal_np[:4] - lam_eigen))

            #print(err[i,j])

    #print(err[:,0])
    err = np.log10(err)
    max_ = np.max(err).round(0)  # = 3
    fig, ax = plt.subplots()
    ax      = sns.heatmap(err, xticklabels=rhoN_list, yticklabels=N_list, vmin=0, vmax=max_)
    ax.invert_yaxis()
    #plt.imshow(err, origin='lower', extent=[20,200,10,100])#, extent=[20,200,10,100])
    plt.title('Logarithmic Mean Error')
    plt.xlabel(r'$\rho_N$')
    plt.ylabel(r'$N$')
    plt.savefig('Results/N_rhoN_heatmap.png')
    plt.show()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Project 2 in FYS4150 - Computational Physics')

 	# Creating mutually exclusive group (only 1 of the arguments allowed at each time)
	# Mutually exclusive arguments must be optional
	group = parser.add_mutually_exclusive_group()
	group.add_argument('-B', '--beam',    action="store_true", help="The buckling beam problem")
	group.add_argument('-Q', '--quantum', action="store_true", help="Quantum mechanics")

	# Optional arguments for input values, default values are set
	parser.add_argument('-e', type=int, nargs='?', default= 1,   help="number of electrons (use 1 or 2)")
	#parser.add_argument('-n', type=int, nargs='?', default= 100, help="dimensionality of matrix")


	# If a mutual exclusive argument is not provided -> help message
	# Maybe: add test for electrons?????
	if len(sys.argv) <= 1:
		sys.argv.append('--help')

	args  = parser.parse_args()

	BucklingBeam       = args.beam
	QuantumMechanics   = args.quantum
	n_electrons        = args.e

	#######################
	optional_values = False
	#######################

	if BucklingBeam:
		print('\nThe buckling beam problem\n')  # exercise 2b

		if optional_values:
			N_list = np.linspace(3, 50, 10).round().astype(int)
			N_iterations(N_list, diag, non_diag)
			sys.exit()

		else:

			N 	   	 = 50             	# matrix dimension
			max_it 	 = 2*N**2     		# max iterations
			rho0   	 = 0            	# rho min
			rhoN   	 = 1           		# rho max
			h      	 = (rhoN-rho0)/N   	# step length (h = rhoN/(N+1))
			diag     = 2/h**2   		# diagonal elements
			non_diag = -1/h**2  		# non-diagonal elements

			A = Toeplitz(N, diag, non_diag)

			# Calculate analytical eigenpairs
			lam_eigen, u_eigen = AnalyticalEigenpairs(N, diag, non_diag)  # first level 1, not 0



	if QuantumMechanics:
		print('\nQuantum dots in 3 dimensions')

		N 	   	 = 75             	# matrix dimension
		max_it 	 = 2*N**2     		# max iterations
		rho0   	 = 0            	# rho min
		rhoN   	 = 7           		# rho max
		h      	 = (rhoN-rho0)/N   	# step length (h = rhoN/(N+1))
		diag     = 2/h**2   		# diagonal elements
		non_diag = -1/h**2  		# non-diagonal elements

		# exercise 2d, 'hydrogen'?
		if n_electrons == 1:
			print('-- one electron\n')

			if optional_values:
				### DOES NOT WORK......(yet!) ###
				rhoN_list = np.linspace(1,10,10).astype(int)
				N_list = np.linspace(25, 250, 10).astype(int)
				N_rho(N_list, rho0, rhoN_list, h, diag, non_diag)
				#sys.exit()

			else:

				rho_values = rho(N, rho0, rhoN, h, electron=1)
				new_diag   = diag+rho_values
				A 		   = Toeplitz(N, new_diag, non_diag)
				lam_eigen  = [3, 7, 11, 15]

		# exercise 2e, 'helium'?
		elif n_electrons == 2:
			print('-- two electrons\n')
			# use N=400 and rhoN = 10  -> h = 0.025 ish
			omega_list = [0.01, 0.25, 0.5, 1, 5]  # input with default list maybe?

			EigVec_lowest = []

			for w in range(len(omega_list)):
				omega = omega_list[w]
				print('omega = ', omega)
				rho_values = rho(N, rho0, rhoN, h, electron=2, w=omega)
				new_diag   = diag+rho_values

				A          = Toeplitz(N, new_diag, non_diag)

				# we should probably switch this out with Jacobi????
				#EigVal_np, EigVec_np, numpy_cpu = EigenPairsNumpy(A)
				# only interested in lowest states
				#EigVec_lowest.append(EigVec_np[:,0])

				# Calculate eigenpairs with Jacobi and sort
				EigVal, EigVec, iterations, cpu = Jacobi(A, N, epsilon=1e-8, max_it=max_it)
				EigVal_Jac, EigVec_Jac          = SortEigenpairs(EigVal, EigVec)

				# only interested in lowest states
				EigVec_lowest.append(EigVec_Jac[:,0])

			EigVec_lowest = np.array(EigVec_lowest)
			labels = [r'$\lambda_0, \omega =0.01$',
					  r'$\lambda_0, \omega =0.25$',
					  r'$\lambda_0, \omega =0.5$',
					  r'$\lambda_0, \omega =1$',
					  r'$\lambda_0, \omega =5$']
			plot_eigenvectors(rho0, rhoN, N, EigVec_lowest**2, labels=labels, save=True, BB=False)
			sys.exit()



	# Calculate eigenpairs with numpy and sort, numpy CPU time
	EigVal_np, EigVec_np, numpy_cpu = EigenPairsNumpy(A)

	# Calculate eigenpairs with Jacobi and sort
	EigVal, EigVec, iterations, cpu = Jacobi(A, N, epsilon=1e-8, max_it=max_it)
	EigVal_Jac, EigVec_Jac          = SortEigenpairs(EigVal, EigVec)

	# Printing the 4 first eigenvalues
	lambda_table = create_df(lam_eigen[:4], EigVal_np[:4], EigVal_Jac[:4])
	print(lambda_table)

	if BucklingBeam:
		FirstEigVec_analytical = u_eigen[1,:]
		FirstEigVec_Jacobi     = EigVec_Jac[:,0]

		SecondEigVec_analytical = u_eigen[2,:]
		SecondEigVec_Jacobi     = EigVec_Jac[:,1]

		eigenvectors = np.array([FirstEigVec_analytical,
								 FirstEigVec_Jacobi,
								 SecondEigVec_analytical,
								 SecondEigVec_Jacobi])
		labels = [r'Analytical $\lambda_0$',
		          r'Jacobi $\lambda_0$',
		          r'Analytical $\lambda_1$',
		          r'Jacobi $\lambda_1$']
		plot_eigenvectors(rho0, rhoN, N, eigenvectors, labels=labels, save=True, BB=True)
