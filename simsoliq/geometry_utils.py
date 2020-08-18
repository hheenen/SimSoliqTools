"""
This module contains helper functions for handling manipulation 
of atom geometries

"""

import numpy as np


def _correct_vec(vec):
    ''' correct vectors in fractional coordinates 
        (assuming vectors minimal connection between 2 points)
    '''
    vec[np.where(vec >= 0.5)] -= 1.0
    vec[np.where(vec < -0.5)] += 1.0
    return(vec)


def find_max_empty_space(atoms, edir=3):
    """
    NOTE: copied from ase-espresso! Python3 compatibility & stand-alone
    Assuming periodic boundary conditions, finds the largest
    continuous segment of free, unoccupied space and returns
    its midpoint in scaled coordinates (0 to 1) in the edir direction (default z).
    """
    position_array = atoms.get_scaled_positions()[..., edir - 1]  # 0-indexed direction
    position_array.sort()
    differences = np.diff(position_array)
    differences = np.append(differences, position_array[0] + 1 - position_array[-1])  # through the PBC
    max_diff_index = np.argmax(differences)
    if max_diff_index == len(position_array) - 1:
        return (position_array[0] + 1 + position_array[-1]) / 2. % 1  # should be < 1 in cell units
    else:
        return (position_array[max_diff_index] + position_array[max_diff_index + 1]) / 2.


def get_CN(atoms, rcut, type_a='*', type_b='*'):
    rpos = atoms.get_scaled_positions(); cell = atoms.get_cell()
    inds = []
    for ty in [type_a,type_b]:
        if ty == '*':
            ty = list(range(len(atoms)))
        else:
            ty = np.array([np.where(atoms.get_atomic_numbers() == t)[0] \
                                                for t in ty]).flatten()
        inds.append(ty)
    cns = []
    for i in range(len(inds[0])):
        cns.append(__get_immediate_CN(rpos[inds[1],:],rpos[i,:],cell,rcut).size - 1)
    return(np.array(inds[0]), np.array(cns))


def __get_immediate_CN(pos_array,pos,cell,rcut):
    ''' function to calculate distance array (pos_array - pos) and determine
        entries within distance rcut
        input:  pos_array = positions which to calculate distances from
                pos       = origin position
                cell      = transformation for distance vectors
                rcut      = cutoff for which to obtain points within distance
        output: cord      = entries of points in pos_array within distance rcut
    '''
    dvec = _correct_vec(pos_array-pos)
    dvec = np.dot(dvec,cell)
    dist = np.linalg.norm(dvec,axis=1)
    cord = np.where(dist <= rcut)[0]
    return(cord)

