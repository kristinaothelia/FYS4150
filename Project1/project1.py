import numpy as np
import matplotlib.pylab as plt
from numba import jit


def forward(n, a, b, c, f, b_tilde, f_tilde):
	'''
	Function for calculating the forward substitution
	'''

	print(n)

	for i in range(n):
		b_tilde[i] = b[i]   - (a[i]*c[i-1])/b_tilde[i-1]
		f_tilde[i] = f[i+1] - (a[i]*f[i-1])/b_tilde[i-1]

	return b_tilde, f_tilde

def backward(n, v, f_tilde, b_tilde):
	'''
	Function for calculating the backward substitution
	'''

	j = n+1

	while j >= 2: 

		v[j-1] = (f_tilde[j-2] - c[j-2]*v[j])/b_tilde[j-2]
		j     -= 1

	return v


def ex_1b(n, a, b, c, f):

	v = np.zeros(n+2)

	b_tilde = np.zeros(n)
	f_tilde = np.zeros(n)


	# boundary conditions
	a[0]       = 0
	c[-1]      = 0
	b_tilde[0] = b[0]
	f_tilde[0] = f[1]


	b_tilde, f_tilde = forward(n, a, b, c, f, b_tilde, f_tilde)

	v 				 = backward(n, v, f_tilde, b_tilde)

	return v


def plot(u, v, x):

	plt.plot(u, x, label='u')
	plt.plot(v, x, label='v')
	plt.legend()
	plt.show()



if __name__ == "__main__":

	n = 1

	h = 1/(n+1)

	# Creating the one-dimensional vectors 
	a = -1*np.ones(n)
	b =  2*np.ones(n)
	c = -1*np.ones(n)


	x = np.linspace(0, 1, n+2)

	f = h*h*100*np.exp(-10*x)

	# Creating the matrix A
	A = 2*np.eye(n) - np.eye(n, k = -1) - np.eye(n, k=1)

	
	# A closed-form solution
	u = 1 - (1 - np.exp(-10))*x - np.exp(-10*x)

	# Calculating the numerical solution
	v = ex_1b(n, a, b, c, f)

	plot(u, v, x)