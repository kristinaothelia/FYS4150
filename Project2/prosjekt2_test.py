"""
Simple tests to check some of the functions in prosjekt2.py
"""

import prosjekt2
import numpy as np

M = np.array(([1,2,3,4,5], [1,2,3,8,5], [1,2,10,4,5], [1,7,3,4,5], [1,2,3,4,5]))

N = len(M)
maxnondiag, k, l = prosjekt2.MaxNonDiag(M, N)
EigVal, EigVec, iterations, cpu = prosjekt2.Jacobi(M, N, epsilon=1e-8, max_it=100)

dot  = []
dots = []

for i in range(len(EigVec)):
	for j in range(len(EigVec)):
		dot.append(EigVec[i].T@EigVec[j])

		if i == j:
			dots.append(1)
		else:
			dots.append(0)

def test_MaxNonDiag():
	maxnondiag, k, l = prosjekt2.MaxNonDiag(M, N)
	solution         = 8

	assert maxnondiag == solution,\
	print('Ikke lik')
	print('Like')


def test_ortho():
	for i in range(len(dot)):
		assert abs(np.array(dot[i]) - np.array(dots[i])) < 1e-8