import numpy as np
from group.check import isSO
from group.rodriguez import hat


def axis_angle(R):
    """ Computes the axis angle representation of a rotation matrix """
    assert(isSO(R))
    A = R - R.transpose()
    if max(abs(A.flatten())) < 1e-8:
        return 0.0, np.zeros(3)
    theta = np.arccos((np.trace(R)-1.0) / 2.0)
    k = np.array([-A[1, 2], A[0, 2], -A[0, 1]])
    k = k / np.linalg.norm(k)
    return theta, k

def quat2rot(q):
    """ converts a quaternion to a rotation matrix """
    assert(abs(np.linalg.norm(q)-1.0) < 1e-7)
    w, x = q[0], q[1:]
    xx = np.outer(x, x)
    return (2*w**2-1)*np.eye(3) + 2*xx + 2*w*hat(x)

def rot2quat(R):
    """ converts a rotation matrix to a quaternion """
    theta, k = axis_angle(R)
    w = np.cos(theta/2.0)
    x = np.sin(theta/2.0) * k
    return np.array([w, x[0], x[1], x[2]])
