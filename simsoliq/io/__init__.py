"""
This module contains a primitive init_mdtraj function which loads the format 
dependent functions and returns the corresponding ``mdtraj`` object

The read-out is not directly processed since reading structures may be 
costly and unnecessary for every case

"""

import os
import numpy as np
from ase.utils import import_module

def init_mdtraj(filename, fmat='vasp', **kwargs):
    """
      function to create a 'mdtraj' (io) object
      uses the appropriate format (only VASP)

      Parameters
      ----------
      filename : str
          path to file
      fmat : str
          format for filetypes to expect
      kwargs : key-value pairs
          further input to pass to mdtraj

      Returns
      -------
      mdtraj : python-object
          object to handle io-routines
      """
    
    fpath = '/'.join(filename.split('/')[:-1])
    fname = filename.split('/')[-1]
    
    # check if file exists - ambiguous filename
    if filename.find('*') == -1 and not os.path.isfile(filename):
        raise IOError("%s does not exist"%filename)

    # case wildcard in output file
    if fname.find('*') != -1 and not os.path.isdir(fpath) and \
        np.any([f.find(fname.split('*')[0]) != -1 for f in listdir(fpath)]):
        raise IOError("%s does not exist"%filename)

    # case wildcard in path
    if fpath.find('*') != -1:
        fsplit = fpath.split('/')
        iwild = [i for i in range(len(fsplit)) if fsplit[i].find('*') != -1][0]
        fpath = '/'.join(fsplit[:iwild])
        if not np.any([os.path.isfile(fpath+'/'+f+'/'+fname) \
            for f in os.listdir(fpath)]):
            raise IOError("%s does not exist"%filename)
    
    # check if file format known
    if fmat not in fmat_dict:
        raise NotImplementedError("%s is an unknown format"%(fmat))
    if fname.split('*')[0] not in fmat_dict[fmat]:
        raise NotImplementedError("%s is unknown to format %s"%(fname, fmat))
    
    # import functions
    module = import_module('simsoliq.io.' + fmat + '_mdtraj')
    read_mdtraj = getattr(module, 'read_mdtraj_' + fmat, None)
    
    # return mdtraj-object
    return(read_mdtraj(fpath, fname, **kwargs))


fmat_dict = {
    'vasp':["vasprun.xml","OUTCAR","vasprun"]
    }



