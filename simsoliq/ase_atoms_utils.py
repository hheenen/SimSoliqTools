"""
This module contains helper functions for handling 
ase-atoms objects

"""

import numpy as np
from ase.atoms import Atoms

def get_type_indices(atoms,atom_types):
    """ 
      helper function to return indeces of atoms in
      a given ASE-Atoms object `atoms` with atom-types
      contained in list `atom_types`
    """
      
    ind_types = np.hstack([np.where(atoms.get_atomic_numbers() == t)[0]\
                for t in atom_types])
    return(ind_types)

def get_type_atoms(atoms,atom_types):
    """ 
      helper function to return new Atoms object from
      given ASE-Atoms object `atoms` only containing
      atoms of type contained in `atom_types`
    """
    ind_types = get_type_indices(atoms,atom_types)
    return(get_inds_atoms(atoms,ind_types))

def get_inds_atoms(atoms,inds):
    """ 
      helper function to return new Atoms object from
      given ASE-Atoms object `atoms` only containing
      atoms with indeces `inds`
    """
    return(Atoms(numbers=atoms.get_atomic_numbers()[inds],\
                 positions=atoms.get_positions()[inds,:],\
                 cell=atoms.get_cell(),
                 tags=atoms.get_tags()[inds]))

