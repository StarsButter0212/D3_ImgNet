#!/usr/bin/env python3
from collections import defaultdict
import os
import sys
import pickle
import numpy as np
from scipy import spatial
from datetime import datetime
from tqdm import tqdm

import preprocess_args as args
from group.generate import rtpairs, generate_rot_matrix

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


"""Dictionary of atomic numbers."""
all_atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
             'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
             'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
             'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
             'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
             'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
             'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
             'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
             'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
             'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
             'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
             'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

atomicnumber_dict = dict(zip(all_atoms, range(1, len(all_atoms)+1)))

basis_dict = {'STO-3G': [1, 1], '6-31G': [6, 4], '6-311G': [6, 5]}


def create_gauss_grid_field(R, n_point, theta, u):
    """Create the gauss grid field of a molecule."""
    trans_coords = []
    r, t = rtpairs(R, n_point)
    x_norm, y_norm, z_norm = np.zeros_like(t), r * np.cos(t), r * np.sin(t)
    coords = np.transpose(np.asarray([x_norm, y_norm, z_norm]))

    thetas = theta * np.arange((np.pi / theta))
    for t in thetas:
        trans_coords.append(np.dot(coords, generate_rot_matrix(t, u)))
    trans_coords = np.concatenate(trans_coords, axis=0)
    trans_coords = np.round(trans_coords, decimals=8)
    trans_coords = np.where(np.abs(trans_coords) > 1e-10, trans_coords, 0)
    trans_coords, index = np.unique(trans_coords, axis=0, return_index=True)
    trans_coords = trans_coords[index.argsort()]
    return trans_coords


def create_sphere(radius_step, radius_min,
                  radius_max, n_points, rot_angle, rot_axis):
    """Create the sphere to be placed on each atom of a molecule."""
    sphere = []
    for n in range(int((radius_max - radius_min) / radius_step + 1)):
        R = radius_min + n * radius_step
        coords = create_gauss_grid_field(R, n_points, rot_angle, rot_axis)
        sphere.append(coords)
    return np.concatenate(sphere)


def create_field(sphere, coords):
    """Create the grid field of a molecule."""
    field = [f for coord in coords for f in sphere+coord]
    return np.array(field)


def create_orbitals(orbitals, orbital_dict):
    """Transform the atomic orbital types (e.g., H1s, C1s, N2s, and O2p)
    into the indices (e.g., H1s=0, C1s=1, N2s=2, and O2p=3) using orbital_dict.
    """
    orbitals = [orbital_dict[o] for o in orbitals]
    return np.array(orbitals)


def create_distancematrix(coords1, coords2):
    """Create the distance matrix from coords1 and coords2."""
    distance_matrix = spatial.distance_matrix(coords1, coords2)
    return np.where(distance_matrix == 0.0, 1e6, distance_matrix)


def create_dataset(dir_dataset, filename, basis_set, radius_min, radius_max,
                   radius_step, n_points, rot_angle, rot_axis, orbital_dict, property=True):

    """Directory of a preprocessed dataset."""
    dir_preprocess = (dir_dataset + 'create_data' + '_' + basis_set +
                      '/' + filename + '_' + basis_set + '/')

    os.makedirs(dir_preprocess, exist_ok=True)

    """Basis set."""
    inner_outer = basis_dict[basis_set]
    inner, outer = inner_outer[0], inner_outer[1]

    """A sphere for creating the grid field of a molecule."""
    sphere = create_sphere(radius_step, radius_min, radius_max,
                           n_points, rot_angle, rot_axis)

    """Load a dataset."""
    with open(dir_dataset + filename + '.txt', 'r') as f:
        dataset = f.read().strip().split('\n\n')

    loop_data = tqdm(enumerate(dataset), total=len(dataset), file=sys.stdout)
    start_time = datetime.now()

    for index, data in loop_data:

        """Index of the molecular data."""
        data = data.strip().split('\n')
        idx = data[0]

        if property:
            atom_xyzs = data[1:-1]
            property_values = data[-1].strip().split()
            property_values = np.array([[float(p) for p in property_values]])
        else:
            atom_xyzs = data[1:]

        atoms = []
        atomic_numbers = []
        N_atoms = 0
        N_electrons = 0
        atomic_coords = []
        atomic_orbitals = []
        orbital_coords = []
        n_quantum_numbers = []
        integral_exponents = []

        """Load the 3D molecular structure data."""
        for atom_xyz in atom_xyzs:
            atom, x, y, z = atom_xyz.split()
            atoms.append(atom)
            atomic_number = atomicnumber_dict[atom]
            atomic_numbers.append([atomic_number])
            N_atoms += 1
            N_electrons += atomic_number
            xyz = [float(v) for v in [x, y, z]]
            atomic_coords.append(xyz)

            """Atomic orbitals (basis functions)
               and principle quantum numbers (q=1,2,...).
            """
            if atomic_number <= 2:
                aqs = [(atom+'1s' + str(i), 1, (0, 0, 0)) for i in range(outer)]
            elif atomic_number >= 3:
                aqs = ([(atom+'1s' + str(i), 1, (0, 0, 0)) for i in range(inner)] +
                       [(atom+'2s' + str(i), 2, (0, 0, 0)) for i in range(outer)] +
                       [(atom+'2p' + str(i), 2, (1, 0, 0)) for i in range(outer)])
            for a, q, l in aqs:
                atomic_orbitals.append(a)
                orbital_coords.append(xyz)
                n_quantum_numbers.append(q)
                integral_exponents.append(l)

        """Create each data with the above defined functions."""
        molecular_formula = ''.join(atoms)
        atomic_coords = np.array(atomic_coords)
        atomic_orbitals = create_orbitals(atomic_orbitals, orbital_dict)
        field_coords = create_field(sphere, atomic_coords)
        distance_matrix = create_distancematrix(field_coords, orbital_coords)
        n_quantum_numbers = np.array([n_quantum_numbers])
        l_quantum_numbers = np.array([np.sum(np.asarray(integral_exponents), axis=1)])
        N_electrons = np.array([[N_electrons]])

        """Save the above set of data."""
        data = [idx,
                N_atoms,
                atomic_orbitals.astype(np.int64),
                distance_matrix.astype(np.float32),
                n_quantum_numbers.astype(np.float32),
                l_quantum_numbers.astype(np.float32),
                N_electrons.astype(np.float32),
                property_values.astype(np.float32),
                molecular_formula]

        data = np.array(data, dtype=object)
        np.save(dir_preprocess + idx, data)

        delta_time = datetime.now() - start_time
        loop_data.set_description('\33[36m【NO. {0:05d}】'.format(index + 1))
        loop_data.set_postfix({'cost_time': '{0}'.format(delta_time)}, '\33[0m')


if __name__ == "__main__":

    """Args."""
    dataset = args.dataset
    basis_set = args.basis_set
    radius_min = args.radius_min
    radius_max = args.radius_max
    radius_step = args.radius_step
    n_points = args.n_points
    rot_angle = args.rot_angle
    rot_axis = args.rot_axis

    """Dataset directory."""
    dir_dataset = '../dataset/' + dataset + '/'

    """Initialize orbital_dict, in which
    each key is an orbital type and each value is its index.
    """
    orbital_dict = defaultdict(lambda: len(orbital_dict))
    print('Preprocess', dataset, 'dataset.\n'
          'The preprocessed dataset is saved in', dir_dataset, 'directory.\n'
          'If the dataset size is large, '
          'it takes a long time and consume storage.\n'
          'Wait for a while...')
    print('-'*50)

    print('Training dataset...')
    create_dataset(dir_dataset, 'train',
                   basis_set, radius_min, radius_max, radius_step,
                   n_points, rot_angle, rot_axis, orbital_dict)
    print('-'*50)

    print('Validation dataset...')
    create_dataset(dir_dataset, 'val',
                   basis_set, radius_min, radius_max, radius_step,
                   n_points, rot_angle, rot_axis, orbital_dict)
    print('-'*50)

    print('Test dataset...')
    create_dataset(dir_dataset, 'test',
                   basis_set, radius_min, radius_max, radius_step,
                   n_points, rot_angle, rot_axis, orbital_dict)
    print('-'*50)

    dir_preprocess = (dir_dataset + 'create_data' + '_' + basis_set + '/')
    with open(dir_preprocess + 'orbitaldict_' + basis_set + '.pickle', 'wb') as f:
        pickle.dump(dict(orbital_dict), f)

    print('The preprocess has finished.')
