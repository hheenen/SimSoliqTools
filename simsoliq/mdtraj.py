"""
This module contains the MDtraj object for data manipulation

"""

import os
import numpy as np
from ase.data import chemical_symbols as symbols
from ase.symbols import Symbols

from simsoliq.io.data_mdtraj import DataMDtraj
from simsoliq.ase_atoms_utils import get_type_indices, get_type_atoms, \
                                     get_inds_atoms
from simsoliq.geometry_utils import _correct_vec
from simsoliq.helper_functions import load_pickle_file, write_pickle_file

# NOTE for documentation needs to be reported that some functions rely on fixed setup --> interface normal to z-coordinate and metal on lowest

class MDtraj(DataMDtraj):
    """ MDtraj object
      
      central object of data-manipulation in simsoliq
      derived from DataMDtraj which manages data-handling
      
      Parameters
      ----------
      See simsoliq.io.data_mdtraj.DataMDtraj for parser
      options to pass
      
      The MDtraj object is most conveniently created via the `init_mdtraj`
      function (simsolig.io.init_mdtraj). The MDtraj object can be used
      for data manipulation and analysis.
      
      Example
      -------
      # creation of mdtraj-object
      from simsoliq.io import init_mdtraj
      mdtraj = init_mdtraj("path_to_mddata", fmat='vasp')

      # retrieving energies
      epot = mdtraj.get_potential_energies()
      ...
    
    """
    def __init__(self, bpath, fident, concmode=None, fnest=None, \
                 safe_asetraj_files=True):
        # set-up DataMDtraj
        DataMDtraj.__init__(self, bpath, fident, \
            concmode=concmode, fnest=fnest, \
            safe_asetraj_files=safe_asetraj_files)
        # other options to follow ...


    def __repr__(self):
        """ 
          basic information for object when printed
        """
        self._retrieve_energy_data()
        outstr = 70*'#' + "\nmdtraj object with %i snapshots\n"%self.mdtrajlen
        outstr += "of composition %s\n"%self.get_traj_composition()
        outstr += "data is located in %s\n"%self.bpath + 70*'#'
        return(outstr)


    def get_traj_energies(self):
        """
          method to return all energy data for an MD trajectory
 
          Parameters
          ----------
          None
 
          Returns
          -------
          edat : dictionay
              dictionary with entries `ekin`, `epot`, `etot`, and meta data
              each holding a 1d numpy array of the kinetic, potential, total
              energy of each snapshot and with the length of the trajectory
        """
        self._retrieve_energy_data()
        return(self.edat)


    def get_kinetic_energies(self):
        """
          method to return kinetic energies for an MD trajectory
 
          Returns
          -------
          edat : 1d numpy array
              array with kinetic energy with length of the trajectory
        """
        self._retrieve_energy_data()
        return(self.edat['ekin'])


    def get_potential_energies(self):
        """
          method to return potential energies for an MD trajectory
 
          Returns
          -------
          edat : 1d numpy array
              array with potential energy with length of the trajectory
        """
        self._retrieve_energy_data()
        return(self.edat['epot'])


    def get_total_energies(self):
        """
          method to return total energies for an MD trajectory
 
          Returns
          -------
          edat : 1d numpy array
              array with total energy with length of the trajectory
        """
        self._retrieve_energy_data()
        return(self.edat['etot'])
    
    
    def get_traj_atoms(self, safe_asetraj_files=True):
        """
          method to return all structures of an MD trajectory 
          as a list of ase-atoms objects
 
          Parameters
          ----------
          safe_asetraj_files : bool
              option to save ase trajectory files of all 
              output files (in their respective directory)
 
          Returns
          -------
          mdtraj_atoms : list
              list of ase-atoms objects of each snapshot of the
              md-trajectory with the length of the trajectory
        
        """
        
        self._retrieve_atom_data(safe_asetraj_files)
        return(self.mdtraj_atoms)


    def get_density_profile(self, height_axis=2, tstart=0, savepkl=True):
        """
          method to return average density profile of trajectory
 
          Parameters
          ----------
          height_axis : int
            axis along which to process the density profile
          tstart : int
            snapshot from which to begin sampling
          savepkl : bool
            save density data into pkl file for faster reading
 
          Returns
          -------
          dens_data : dictionary
            included: `binc` center of bins of density, `hists` histograms of 
            density for each element and separated solvent (tag=solv) 
            in mol/cm3
        
        """
        
        pklfile = self.bpath+'/mdtraj_density.pkl'
        # process density
        if not savepkl or not os.path.isfile(pklfile):
            dens_data = self._get_density_profile(height_axis, tstart)
        # read density
        elif savepkl:
            dens_data = load_pickle_file(pklfile)
        # save density
        if savepkl:
            write_pickle_file(pklfile, dens_data)

        return(dens_data)


    def _get_density_profile(self, height_axis, tstart):
        """ 
          function to process densities - see arguments in 
          `get_density_profile`
          (only tested and used for water)
        
        """
        solv = self._get_solvent_indices(snapshot=0)
        traj = self.mdtraj_atoms
 
        # indices of solvent 
        ind_c = list(solv.keys())
        ind_l = list(np.array(list(solv.values())).flatten())
        # symbols for solvent
        sym_c = '-'.join([symbols[c] for c in \
            np.unique(traj[0].get_atomic_numbers()[ind_c])])+'solv'
        sym_l = '-'.join([symbols[c] for c in \
            np.unique(traj[0].get_atomic_numbers()[ind_l])])+'solv'
        
        # indices of others
        atypes = np.unique(traj[0].get_atomic_numbers())
        ind_o = []
        for atype in atypes:
            inner = []
            for i in np.where(traj[0].get_atomic_numbers() == atype)[0]:
                if i not in ind_c+ind_l:
                    inner.append(i)
            ind_o.append(inner)
                    
        inds = [ind_c,ind_l] + ind_o
        
        # cut traj here
        traj = traj[int(tstart):]

        # compute histogram
        ha = height_axis
        bins = np.linspace(0,traj[0].get_cell()[ha,ha],200) # hist by height
        hists = [np.zeros(len(bins)-1) for i in range(len(inds))]
        for snap in traj:
            pos = snap.get_positions()
            for i in range(0,len(inds)):
                d = np.histogram(pos[inds[i],ha],bins)
                hists[i] += d[0]
        for i in range(0,len(hists)): # normalize time
            hists[i] /= len(traj) 
 
        # convert histogram into density
        dA = traj[0].get_volume()/traj[0].cell[ha,ha]
        dh = bins[1] - bins[0]
        f_convert = 1.660538921e-24/(dA*dh*(1e-8)**3) #conversion mol/cm3
        for i in range(0,len(hists)):
            hists[i] *= f_convert
        
        # make dict for output
        dens_types = [sym_c,sym_l]+[symbols[a] for a in atypes]
        hist_dicts = {dens_types[i]:hists[i] for i in range(len(dens_types))
            if hists[i].sum() > 0}
        
        # bin center
        binc = (bins[0:-1] + bins[1:])/2
 
        return({'binc':binc,'hists':hist_dicts})


    def _get_solvent_indices(self, snapshot=0, smol=[8,2], rcut=1.3):
        """
          method to return structured solvent indices of a snapshot
          (only tested and run for water)
 
          Parameters
          ----------
          snapshot : int
            integer of the snapshot to evaluate
          smol : list (length=2)
            identifier with [0] type of central solvent atom 
                            [1] number of coordinating atoms
          rcut : float
            cutoff to use between central and coordinating atoms
 
          Returns
          -------
          solv : dict
              dict including each central atom as a key and 
              coordinating atoms as value which fulfill criteria
        
        """
        # NOTE: could be robuster by including type of coordinating
        #       ion and find a better method than rcut
        self._retrieve_atom_data(self.safe_asetraj_files)
        atoms = self.mdtraj_atoms[snapshot]
        
        pos = atoms.get_scaled_positions()
        cell = atoms.get_cell()
        
        # central atom indices
        ind_O = get_type_indices(atoms,[smol[0]])
        # other atom indices
        type_res = list(set((atoms.get_atomic_numbers())))
        type_res.remove(smol[0])
        ind_rest = get_type_indices(atoms,type_res)
        type_R = atoms.get_atomic_numbers()[ind_rest]
        pos_R = pos[ind_rest,:]
        
        # check neighbors in all types 
        solv = {}
        for o in ind_O:
            pos_O = pos[o,:]
            d_vec = np.linalg.norm(np.dot(_correct_vec(pos_R - pos_O),cell),axis=1)
            ind_neigh = np.where(d_vec < 1.3)[0]
            if sum(type_R[ind_neigh]) == smol[1]:
                solv.update({o:ind_rest[ind_neigh]})
        return(solv)


    def get_traj_composition(self):
        """
          method to return a string of the atomic composition of the trajectory
        
        """
        # get atomic data
        traj0 = self.get_first_snapshot()
        atno = traj0.get_atomic_numbers()
        
        # get solvent indices
        solv = self._get_solvent_indices(snapshot=0)
        assert np.all([len(solv[k])==2 for k in solv])
        ind_solv = list(solv.keys()) + list(np.array(list(solv.values())).flatten())

        # get substrate indices
        type_subs = self.get_substrate_types()
        ind_subs = [i for i in range(len(traj0)) if atno[i] in type_subs \
            and i not in ind_solv]
            
        # get other inds assumed as adsorbate
        ind_ads = [i for i in range(len(traj0)) if i not in ind_subs+ind_solv]

        # prep composition string
        sym = Symbols([atno[i] for i in ind_subs])
        str_subs = sym.get_chemical_formula()
        sym = Symbols([atno[i] for i in ind_ads])
        str_ads = sym.get_chemical_formula()
        str_solv = "%iH2O"%len(solv) #hardcoded to water right now

        str_comp = "_".join([k for k in [str_subs, str_solv, str_ads]\
            if len(k) > 0])
        return(str_comp)


    def get_substrate_types(self):
        """
          method to return the (assumed) types of the substrate
 
          Parameters
          ----------
          None
 
          Returns
          -------
          type_subs : list
              list including elemental numbers of substrate
        
        """
        # NOTE: get substrate indices: is assumed to be fixed or 
        #       lowest z-coordinate
        traj0 = self.get_first_snapshot()
        atno = traj0.get_atomic_numbers()
        
        if len(traj0.constraints) > 0:
            ind_cnst = traj0.constraints[0].index
            type_subs = np.unique(atno[ind_cnst])
        else: # not tested
            pos = traj0.get_positions(); ind_zmin = pos[:,2].argmin()
            type_subs = [atno[ind_zmin]]
        
        return(type_subs)


