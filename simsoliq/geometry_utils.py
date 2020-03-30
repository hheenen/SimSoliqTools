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
