"""
This module contains functionality for data management of 
the mdtraj object

"""

import os
import numpy as np
from ase.io import read, write
from simsoliq.io.utils import _nested_iterator, _level_iterator, \
    _level_nested_iterator
from simsoliq.helper_functions import load_pickle_file, write_pickle_file


# TODO: _iterator function can probably be made more elegant with a decorator
#       currently does not allow to pass option 'safe_traj_file' for example
#       ghost option (always turned on)
# TODO: move safe_traj_file to this level/iterator level of the code!!!
#       cleaner and better to handle

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
        self.sp_prep = None
        self.efunc_sp = None
        self.fefunc_sp = None
        self.vacefunc_sp = None
        self.wffunc_sp = None
        self.chgfunc_sp = None
        self.elpot_sp = None

        # handling of file-iterators
        self.concmode = concmode
        if concmode in ['nested', 'level_nested'] and fnest == None:
            raise ValueError("need to define fnest for `nested` concmodes")
        if concmode == 'nested' and fident.find('*') != -1:
            raise ValueError("conmode='nested' requires unambigous fident")
        self.fnest = fnest

        if fident.find('*') != -1:
            self.concmode = 'level'


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
            only invoced if no self.mdtraj_atoms object present
            read-out usually takes long
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
        
        elif self.concmode == 'level_nested':
            return(_level_nested_iterator(efunc,\
                self.bpath, self.fident, self.fnest))

        else:
            raise NotImplementedError("unknown concmode")


    def prepare_singlepoint_calculations(self, tag='', freq=500, spatoms=None, **kwargs):
        """
          method to prepare single point calculations on snapshots of the 
          trajectory. This can be usefull if data relying on heavy calculations
          is needed and can only be extract sampling few snapshots
 
          Parameters
          ----------
          tag : str
            tag for the singlepoint calculation folder singlepoints_`tag`
          freq : float
            frequency of snapshots used to prepare a singlepoint calculation
          spatoms : list of ase-atoms objects (optional)
            if manipulated atoms are to be used can be handed here

          Returns
          -------
          cpaths : list
            list including all directories which were prepared
        """
        # prepare path for singlepoints
        spname = 'singlepoints'
        if len(tag) != 0:
            spname = '_'.join([spname, tag])
        sppath = self.bpath+'/'+ spname
        if not os.path.isdir(sppath):
            os.mkdir(sppath)

        # get atoms 
        if spatoms == None:
            self._retrieve_atom_data()
            spatoms = self.mdtraj_atoms
            # tag atoms - this may be relevant to pseudopotentials
            for i in range(len(spatoms)):
                spatoms[i].set_tags(np.arange(len(spatoms[i]))) 
        else:
            assert len(spatoms) == self.mdtrajlen

        # prepare folders for freq
        cpaths = []
        for i in range(0,self.mdtrajlen,freq):
            cdir = sppath+'/sp_step{0:05d}'.format(i)
            if not os.path.isdir(cdir):
                os.mkdir(cdir)
            prep = self.sp_prep(cdir, self.bpath, spatoms[i], **kwargs)
            if prep:
                cpaths.append(cdir.split('/')[-1])
        return(cpaths)
 

    def read_singlepoint_calculations(self, dkey, tag='', \
        safe_pkl_files=True, timesteps=[], dfunc=None, **kwargs):
        """
          method to read single point calculations on snapshots of the 
          trajectory. 
 
          Parameters
          ----------
          tag : str
            tag for the singlepoint calculation folder singlepoints_`tag`
          dkey : str
            key to determine function for data read-out
            options include: 'epot', 'efermi', 'evac', 'wf', 'dchg', 'elpot'
          timesteps : list, optional
            list of timesteps which are read out, if empty all are read
          dfunc : function, optional
            function to read out data from singlepoint
            not needed if standard key is invoked

          Returns
          -------
          sp_data : dict
            dictionary containing sp data with {timestep:data}
            dictionary is chosen as data could have any form
        """
        # get sp_path directory
        sp_path = self._get_sp_path(tag=tag)

        # find subfolders in `singlepoints`
        sp_sub = [f for f in os.listdir(sp_path) \
            if os.path.isdir(sp_path+'/'+f)]

        # setup timesteps if none given
        if len(timesteps) == 0:
            timesteps = [int(s.split('_')[1][4:]) for s in sp_sub]

        # choose singlepoint data-retrieval function
        fspdict = {'epot':self.efunc_sp, 'efermi':self.fefunc_sp, \
            'evac':self.vacefunc_sp, 'wf':self.wffunc_sp, \
            'dchg':self.chgfunc_sp, 'elpot':self.elpot_sp}
        if dkey in fspdict:
            dfunc = fspdict[dkey]
        elif dfunc == None:
            raise NotImplementedError("unknown dkey")

        # find previously stored data
        pklfile = sp_path+'/%ss.pkl'%dkey
        sp_data = {}
        if safe_pkl_files and os.path.isfile(pklfile):
            sp_data = load_pickle_file(pklfile)
        
        # iterate singlepoint folders
        for sub in sp_sub:
            td = int(sub.split('_')[1][4:])
            if td not in sp_data and td in timesteps:
                dat = dfunc(sp_path+'/'+sub, **kwargs)
                sp_data.update({td:dat})
        if safe_pkl_files:
            write_pickle_file(pklfile, sp_data)
        return(sp_data)


    def _get_sp_path(self,tag=''):
        """ 
          helper function to find and retrieve singleploints directory
        
        """
        # find "singlepoints" folders
        sp_base = [f for f in os.listdir(self.bpath) \
            if f[:12] == 'singlepoints' and os.path.isdir(self.bpath+'/'+f)]
        if len(sp_base) == 0:
            raise IOError("no `singlepoints` folder found")
        # if missing tag look for uniquely identifyable singlepoint folder
        if len(tag) == 0:
            if len(sp_base) == 1:
                tag = sp_base[0].split('_')[1]
            else:
                raise IOError("please provide tag argument")
        sp_path = self.bpath+'/'+'singlepoints_%s'%tag
        return(sp_path)


    def get_single_snapshot(self, n=0):
        """
          method to return the first snapshot of a trajectory;
          this method circumvents reading the whole trajectory and is
          therefore considerably faster if the time evolution is not needed
 
          Parameters
          ----------
          n : int / str
            number of snapshot to retrieve, can also be a string for 
            slicing, i.e. '::100'

          Returns
          -------
          traj0 : ase-atoms object
            nth snapshot of trajectory

        """
        if len(self.mdtraj_atoms) == 0:
            fatoms = [f for f in os.listdir(self.bpath) \
                if f[:12] == 'mdtraj_atoms']
            fatoms.sort()
            if len(fatoms) > 0:
                try:
                    return(read(self.bpath+'/'+fatoms[0],n))
                except StopIteration:
                    pass
        self._retrieve_atom_data()
        return(self.mdtraj_atoms[n])



