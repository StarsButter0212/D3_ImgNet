import numpy as np


def isSO(R, tol=1e-6):
    error = np.dot(R, R.transpose()) - np.eye(3)
    t1 = max(abs(error.flatten())) < tol
    t2 = abs(np.linalg.det(R)-1.0) < tol**(1.0/3.0)
    return t1 and t2


def isAntiSym(omega, tol=1e-6):
    error = omega + omega.transpose()
    return max(abs(error.flatten())) < tol
