"""
This module contains functionality for creating an mdtraj-object
capable of reading VASP formats

(for later) This module contains functions for reading vasp
input to be inherited by an mdtraj object

"""

import os, subprocess
import numpy as np
from ase.io import read, write
from simsoliq.mdtraj import MDtraj

def read_mdtraj_vasp(fpath, fname, **kwargs):
    """
      function to create a 'vasp-mdtraj' (io) object
      passes arguments to object (see function io.read_mdtraj)

    """

    mdtraj = MDtraj(bpath=fpath, fident=fname, **kwargs)
    mdtraj.efunc = _read_energies_vasp
    mdtraj.afunc = _read_vasp_atoms
    mdtraj.tfunc = _read_vasp_timedata

    return(mdtraj)



############## vasp specific functions to read output files ###################

def _read_energies_vasp(wpath, fident):
    """ 
      function to identify reading data from OUTCAR or vasprun.xml
    
    """
    if fident.find("OUTCAR") != -1:
        return(_read_OUTCAR_energetics(wpath+"/"+fident))
    elif np.all([fident.find(k) != -1 for k in ["vasprun","xml"]]):
        return(_read_xml_energetics(wpath+"/"+fident))


def _read_xml_energetics(filename):
    """
      hard-coded parser for energies in xml files
      TODO: replace with xml parser
    
    """
    process = subprocess.Popen("grep '%s' -B 4 %s"%('"kinetic"',filename),\
                           shell=True,stdout=subprocess.PIPE)
    tstr = process.communicate()[0].decode('utf-8').split()
    #tstr = os.popen("grep '%s' -B 4 %s"%("kinetic",filename)).read().split()
    epot = np.array([float(s) for s in tstr[3::18]])
    ekin = np.array([float(s) for s in tstr[15::18]])
    etot = ekin+epot
    return({'ekin':ekin,'epot':epot,'etot':etot,'runlengths':[etot.size]})


def _read_OUTCAR_energetics(filename):
    """ 
      hard-coded parser for energies in OUTCAR files
      TODO: replace with more reliable code
    
    """
    # Thermodynamics
    ekin = _get_val_OUTCAR(filename,'kinetic energy EKIN   =',4,5)
    epot = _get_val_OUTCAR(filename,'ion-electron   TOTEN  =',4,7)
    etot = ekin+epot
    return({'ekin':ekin,'epot':epot,'etot':etot,'runlengths':[etot.size]})


def _get_val_OUTCAR(filename,string,i,j):
    """ 
      primitive helper function
      TODO: replace with more reliable code
    
    """
    process = subprocess.Popen("grep '%s' %s"%(string,filename),\
                           shell=True,stdout=subprocess.PIPE)
    tstr = process.communicate()[0].decode('utf-8').split()
    
    val = [float(s) for s in tstr[i::j]]
    return(np.array(val))


def _read_vasp_atoms(wpath, fident, safe_asetraj_files=True):
    """
      function to read all structures of a vasp-output file
      as a list of ase-atoms objects
 
      Parameters
      ----------
      wpath : str
          name of path in which output file lies
      fident : str
          name / identifier of file to read
      safe_asetraj_file : bool
          boolean to decide whether simsoliq shall save/use
          trajectory files of output data for each output file
 
      Returns
      -------
      atoms : list of ase-atoms objects
          list of ase-atoms objects of each snapshot of the output file
    
    """
    
    savedfile = wpath+'/mdtraj_atoms_%s.traj'%(fident.split('.')[0].lower())
    
    # no saved file existing
    if not os.path.isfile(savedfile) or \
        os.path.getmtime(savedfile) < os.path.getmtime(wpath+'/'+fident):
        atoms = read(wpath+'/'+fident,':')
        if safe_asetraj_files:
            # save positional data
            write(savedfile, atoms)
    else:
        atoms = read(savedfile,':')
    return(atoms)
    
    
def _read_vasp_timedata(wpath, fident):
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
    #TODO: needs proper implementation hard-coded    
    timedat = {"timeunit":['fs'], "timestep":[1]}
    return(timedat)


