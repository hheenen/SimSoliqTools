"""
This module contains functions to average mdtraj-object data

"""

from copy import deepcopy
import numpy as np

from ase.data import chemical_symbols as symbols


def average_energies(mdtraj_list, tstart):
    """
      function to compute averages of a selection of mdtraj objects
      sorted by their composition
 
      Parameters
      ----------
      mdtraj_list : list
        list of mdtraj objects
      tstart : int
        time after when to evalute (in time units of mdtraj)
 
      Returns
      -------
      edat : dictionay
          dictionary with entries `ekin`, `epot`, `etot` and each composition
          including the average and standard deviation
          
    """
    # get trajectory composition and bundle trajectories
    ekeys = ['etot', 'epot', 'ekin']
    comps = {traj.get_traj_composition():[] for traj in mdtraj_list}
    edat = {ek:deepcopy(comps) for ek in ekeys}
    
    for traj in mdtraj_list:
        comp = traj.get_traj_composition()
        ed = traj.get_traj_energies()
        for ek in ekeys:
            edat[ek][comp].append(np.mean(ed[ek][tstart:]))

    for ek in edat:
        for comp in edat[ek]:
            edat[ek][comp] = {'mean':np.mean(edat[ek][comp]), \
                'std':np.std(edat[ek][comp])/np.sqrt(len(edat[ek][comp]))}
    return(edat)


def sort_energies(mdtraj_list):
    """
      function to sort energy data and return it sorted by composition
 
      Parameters
      ----------
      mdtraj_list : list
        list of mdtraj objects
 
      Returns
      -------
      edat : dictionay
          dictionary with with entries `ekin`, `epot`, `etot` per composition
    """
    ekeys = ['etot', 'epot', 'ekin']
    comps = {traj.get_traj_composition():[] for traj in mdtraj_list}
    edat = {ek:deepcopy(comps) for ek in ekeys}
    timedat = {'timestep':[],'timeunit':[]}

    for traj in mdtraj_list:
        comp = traj.get_traj_composition()
        ed = traj.get_traj_energies()
        for ek in ekeys:
            edat[ek][comp].append(ed[ek])
       # store time data
        for k in timedat:
            timedat[k].append(ed[k])

    # check consistency in time data:
    for k in timedat:
        assert len(list(set(timedat[k]))) == 1
    for k in timedat:
        edat.update({k:list(set(timedat[k]))[0]})

    return(edat)


def average_densities(mdtraj_list, height_axis=2, tstart=0):
    """
      function to average density profiles. Densities are aligned
      to end of substrate
 
      Parameters
      ----------
      mdtraj_list : list
        list of mdtraj objects
      height_axis : int
        axis along which to process the density profile
      tstart : int
        time after when to evalute (in time units of mdtraj)
 
      Returns
      -------
      dens_data : dictionary
        included: `binc` center of bins of density, `hists` histograms of 
        density for each element and separated solvent (tag=solv) 
        in mol/cm3
          
    """
    # sort mdtraj_list by composition
    comps = [traj.get_traj_composition() for traj in mdtraj_list]
    comp_dict= {c:[mdtraj_list[i] for i in range(len(mdtraj_list)) \
        if comps[i] == c] for c in set(comps)}

    comp_dens = {}
    # compute densities for each composition and average
    for comp in comp_dict:
        av_dens = {'hists':{}, 'binc':[]}
        for mdsub in comp_dict[comp]:
            # get inividual density
            dens = mdsub.get_density_profile(height_axis=height_axis, \
                tstart=tstart, savepkl=True)
            # determine substrate species
            esub = [symbols[i] for i in mdsub.get_substrate_types()]
            if len(esub) > 1:
                raise NotImplementedError("multi-elemental substrate found")
            dens = align_density_to_substrate(dens, esub[0])
            
            # update av_dens with bin centers
            if len(av_dens['binc']) == 0:
                av_dens['binc'] = deepcopy(dens['binc'])
            
            # sum-up values - first profile determines shape
            for k in dens['hists'].keys():
                if k not in av_dens['hists']:
                    av_dens['hists'].update({k:np.zeros(dens['hists'][k].shape)})
                lk = min(dens['hists'][k].size, av_dens['hists'][k].size)
                av_dens['hists'][k][:lk] += dens['hists'][k][:lk]

        # average by number of summed up histograms
        for k in av_dens['hists']:
            av_dens['hists'][k] /= len(comp_dict[comp])
        comp_dens.update({comp:av_dens})
    return(comp_dens)


def align_density_to_substrate(dens, esub):
    """
      helper function to align density histogram including substrate
    """
    # find where substrate is in density profile
    indm = np.where(dens['hists'][esub] != 0.0)[0]
    dens['binc'] = dens['binc'][indm[0]:]
    for k in dens['hists']:
        dens['hists'][k] = dens['hists'][k][indm[0]:]
    return(dens)


