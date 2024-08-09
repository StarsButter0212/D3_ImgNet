import numpy as np
from group.check import isSO
from group.check import isAntiSym


def exp(omega):
    """ Computes the exponential map of an anti-symetric matrix or 3-vector """
    if omega.size == 3:
        omega = hat(omega)
    assert(isAntiSym(omega))
    theta = np.sqrt(omega[0,1]**2 + omega[0,2]**2 + omega[1,2]**2)
    if theta < 1e-6:
        return np.eye(3) + omega + 0.5*np.dot(omega,omega) + (1/6.0)*np.dot(omega,np.dot(omega,omega))
    K = omega / theta # division by zero error?
    return np.eye(3) + np.sin(theta)*K + (1.0-np.cos(theta))*np.dot(K,K)

def log(R, return_as_vector=False):
    """ Computes the inverse of the exponential in a ball around the origin """
    assert(isSO(R))
    A = R - R.transpose()
    if max(abs(A.flatten())) < 1e-6:
        if return_as_vector:
            return unhat(A)
        return A
    theta = np.arccos((np.trace(R)-1.0)/2.0)
    omega = (theta / (2*np.sin(theta))) * A
    if return_as_vector:
        return unhat(omega)
    return omega

def hat(k):
    """ Converts a 3-vector to an anti-sym matrix """
    A = np.zeros((3, 3))
    A[0, 1] = -k[2]
    A[0, 2] = k[1]
    A[1, 2] = -k[0]
    A = A - A.transpose()
    return A

def unhat(A):
    """ Inverts the hat-map """
    import numpy as np
    return np.array([-A[1,2], A[0,2], -A[0,1]])
