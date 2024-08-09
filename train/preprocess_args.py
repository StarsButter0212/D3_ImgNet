#!/usr/bin/env bash
import numpy as np

# Dataset.
# dataset = 'QM9under7atoms_atomizationenergy_eV'
dataset = 'QM9under14atoms_atomizationenergy_eV'
# dataset = 'QM9over15atoms_atomizationenergy_eV'

# Basis set.
basis_set = '6-311G'                # '6-311G.gbs'

# Grid field.
radius_min = 0.3
radius_max = 0.4
radius_step = 0.1
n_points = 14                       # one circle's points
n_theta = 4
rot_angle = np.pi/n_theta
rot_axis = [0, 0, 1]               # z-axis




