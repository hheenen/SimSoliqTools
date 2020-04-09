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
