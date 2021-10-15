"""
This module contains functionality for creating an mdtraj-object
out of a ase-trajectory file (simosoliq uses ase natively, only
placeholder functions)

Functions of this module are inherited by an mdtraj object

"""

import os, subprocess
import numpy as np
from copy import deepcopy
from shutil import copyfile
from ase.io import read, write
#from ase.build import sort
#from ase.calculators import vasp
from simsoliq.mdtraj import MDtraj
#from simsoliq.io.utils import _unpack_files, lines_key
#from simsoliq.geometry_utils import find_max_empty_space

# TODO: delete if really not needed


def read_mdtraj_ase(fpath, fname, **kwargs):
    """
      function to create a 'vasp-mdtraj' (io) object
      passes arguments to object (see function io.read_mdtraj)

    """

    mdtraj = MDtraj(bpath=fpath, fident=fname, **kwargs)
    mdtraj.efunc = _read_energies_asetraj
    mdtraj.afunc = _read_atoms_asetraj
    mdtraj.tfunc = _read_timedata_asetraj
    mdtraj.sp_prep = _nofunc
    mdtraj.efunc_sp =  _nofunc
    mdtraj.fefunc_sp =  _nofunc
    mdtraj.vacefunc_sp =  _nofunc
    mdtraj.wffunc_sp =  _nofunc
    mdtraj.chgfunc_sp =  _nofunc
    mdtraj.elpot_sp =  _nofunc

    return(mdtraj)


############## ase specific functions to read output files ###################

def _nofunc():
    raise NotImplementedError("function not implemented")


def _read_energies_asetraj(wpath, fident):
    """ 
      function to read energies from ase-traj object
    
    """
    atoms = read(wpath+'/'+fident, ":")
    epot = np.array([a.get_potential_energy() for a in atoms])
    ekin = np.array([a.get_kinetic_energy() for a in atoms])
    etot = ekin+epot
    return({'ekin':ekin,'epot':epot,'etot':etot,'runlengths':[etot.size]})


def _read_atoms_asetraj(wpath, fident):
    """
      function to read all structures of a vasp-output file
      as a list of ase-atoms objects
 
      Parameters
      ----------
      wpath : str
          name of path in which output file lies
      fident : str
          name / identifier of file to read
 
      Returns
      -------
      atoms : list of ase-atoms objects
          list of ase-atoms objects of each snapshot of the output file
    
    """
    # no saved file existing
    atoms = read(wpath+'/'+fident,':')
    return(atoms)
    
    
def _read_timedata_asetraj(wpath, fident):
    """
      function to read meta data for timestep used in trajectory
 
      Parameters
      ----------
      wpath : str
          name of path in which output file lies
      fident : str
          name / identifier of file to read
 
      Returns
      -------
      timedat : dict
          includes "timeunit" and "timestep"
    
    """
    #TODO: hard-coded, not sure if saved somewhere in traj-file
    timedat = {"timeunit":['fs'], "timestep":[1]}
    return(timedat)

