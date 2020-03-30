"""
This module contains functionality for data management of 
the mdtraj object

"""


import numpy as np
from ase.io import read, write
from simsoliq.io.utils import _nested_iterator, _level_iterator


# TODO: _iterator function can probably be made more elegant with a decorator
#       currently does not allow to pass option 'safe_traj_file' for example
#       ghost option (always turned on)

class DataMDtraj(object):
    """ DataMDtraj object
      
      central object of data-handling simsoliq for data, io
      
      Parameters
      ----------
      bpath : str
          path to md simulation
      fident : str
          filename or identifier of output-file to read
      concmode : None/str, optional
          option for iteration of folders/files of output data
          None (default) - do not look for further output other than
                           bpath+'/'+fident
          `nested` - look for further output in nested folders
                     within `bpath` which are identified via `fnest`
          `level`  - look for further output in `bpath` following wildcard
                     character `*` in `fident`. Automatically invoked if
                     wildcard is found in `fident`.
      fnest : str, optional
          only used in combination with concmode=`nested`
      safe_asetraj_files : bool
          option to determine if an ase-traj file of the trajectory is saved
          within `bpath` (and nested folders). This allows for considerably
          faster i/o.
      
    """
    def __init__(self, bpath, fident, concmode=None, fnest=None, \
                 safe_asetraj_files=True):
        self.bpath = bpath
        self.fident = fident
        self.safe_asetraj_files = safe_asetraj_files
        self.edat = {}
        self.mdtraj_atoms = []
        self.mdtrajlen = 0
        self.timestep = 0
        self.timeunit = "None"
        
        # functions for reading data, possibly better way
        # to handle this, like inheriting them (but may need to be situational)
        self.efunc = None
        self.afunc = None
        self.tfunc = None

        # handling of file-iterators
        self.concmode = concmode
        if concmode == 'nested' and fnest == None:
            raise ValueError("need to define fnest for concmode='nested")
        if concmode == 'nested' and fident.find('*') != -1:
            raise ValueError("conmode='nested' requires unambigous fident")
        self.fnest = fnest

        if fident.find('*') != -1:
            self.concmode = 'level'


    def __repr__(self):
        """ basic information for object when printed
        """
        self._retrieve_energy_data()
        outstr = 70*'#' + "\nmdtraj object with %i snapshots\n"%self.mdtrajlen
        outstr += "data is located in %s\n"%self.bpath + 70*'#'
        return(outstr)


    def _retrieve_energy_data(self):
        """ helper function to run io routines on energies,
            saves data internally in `edat` and `mdtrajlen`
        """
        self._initialize_timestep()
        if len(self.edat) == 0:
            self.edat = self._iterator(self.efunc)
            # add for output information
            self.edat.update({'timestep':self.timestep, \
                'timeunit':self.timeunit})
        if self.mdtrajlen == 0:
            self.mdtrajlen = self.edat['epot'].size
        else:
            assert self.mdtrajlen == self.edat['epot'].size


    def _retrieve_atom_data(self, safe_asetraj_files=True):
        """ helper function to run io routines on atomic positions
            saves data internally in `mdtraj_atoms` and `mdtrajlen`
        """
        # NOTE safe_asetraj_files is a dead option, cannot be 
        #      passed down in current module structure
        
        if len(self.mdtraj_atoms) == 0:
            self.mdtraj_atoms = self._iterator(self.afunc)
        if self.mdtrajlen == 0:
            self.mdtrajlen = len(self.mdtraj_atoms)
        else:
            #print(self.mdtrajlen, len(self.mdtraj_atoms))
            assert self.mdtrajlen == len(self.mdtraj_atoms)


    def _initialize_timestep(self):
        """ helper function to obtain timestep of 
            trajectory data (in fs)
        """
        if self.timestep == 0 and self.timeunit == "None":
            timedat = self._iterator(self.tfunc)
            
            # each partial trajectory should have same timestep
            timedat = {k:list(set(timedat[k])) for k in timedat}
            
            # assert uniform data
            assert np.all([len(timedat[k]) == 1 for k in timedat])
            
            self.timestep = timedat['timestep'][0]
            self.timeunit = timedat['timeunit'][0]


    def _iterator(self,efunc):
        """ helper function to choose iterator return
            (TODO: replace with elegant decorators)
        """
        if self.concmode == None:
            return(efunc(self.bpath, self.fident))

        elif self.concmode == 'nested':
            return(_nested_iterator(efunc, \
                self.bpath, self.fident, self.fnest))

        elif self.concmode == 'level':
            return(_level_iterator(efunc,\
                self.bpath, self.fident))

        else:
            raise NotImplementedError("unknown concmode")


