import numpy as np
from group.quaternion import quat2rot


def generate_rot_matrix(theta, u):
    """ generate a rotation wrt the rotation angle theta and axis u """
    u = np.asarray(u)
    q = np.append(np.cos(theta/2), np.sin(theta/2)*u)
    q = q / np.linalg.norm(q)
    return quat2rot(q)


def rtpairs(r, n):
    r_list, p_list = [], []
    for i in range(n):
        r_list.append(r)
        p_list.append((2*i+1) * (np.pi/n))
    return r_list, p_list
