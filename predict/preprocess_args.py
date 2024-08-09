#!/usr/bin/env bash
import numpy as np

# Dataset used in pre-training.
dataset_trained = 'QM9under14atoms_atomizationenergy_eV'

# Dataset for prediction.
dataset_predict = 'QM9over15atoms_atomizationenergy_eV'  # Extrapolation.


# Basis set.
basis_set = '6-311G'

# Grid field.
radius_min = 0.3
radius_max = 0.4
radius_step = 0.1
n_points = 14                       # one circle's points
n_theta = 4
rot_angle = np.pi/n_theta
rot_axis = [0, 0, 1]               # z-axis



