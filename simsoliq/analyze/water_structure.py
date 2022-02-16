"""
This module contains functions to analyze the time-propagation
of the solvent (and adsorbate) of an mdtraj object

Adsorption is counted partitioned to surface sites; 
reactions with possibly present adsorbates or among water
molecules are tracked and can be filtered if wished outside
of statistics

"""

import numpy as np
from copy import deepcopy
from simsoliq.helper_functions import _arr2hash
from simsoliq.analyze.sampling_sites import get_top_slab_atoms
from simsoliq.geometry_utils import _correct_vec
from itertools import groupby
from operator import itemgetter


    # TODO: 
    # (x) comment decorator + _get_time_average_solvent
    # (x) decorator into misc
    # (x) make unit test for this solvent indices
    # (x) replace (and test) other pkl functions with decorator
    # (x) continue with putting next snipped into analysis package
    # (x) make function to compute snipped analysis for mdtraj object
    # (x) make separate module for site-creation
    # (x) make function H2O adsorption
    # (9) make function for adsorbate tracking !!!!


    # NOTE: eventual split-up of this module if applicable

def partition_solvent_composition(md_traj):
    # TODO: add output for this
    """
      parition trajectory into snippets of constant solvent composition
      this is particularly important of trajectories where adsorbates
      exchange atoms with the solvent (i.e. *OH + H2O --> H2O + *OH)
 
      Parameters
      ----------
      md_traj : mdtraj object
 
      Returns
      -------
      out_sol : list
        list of dictionaries with composition and time-inds of composition
    
    """
    ma_ind = md_traj._get_time_average_solvent_indices()

    # sample snippets of constant solvent composition
    nsolv, nfreq = np.unique(ma_ind.sum(axis=1),return_counts=True)
    out_sol = []
    for ns in nsolv: #iterate changing solvent composition
        ind = np.where(ma_ind.sum(axis=1) == ns)[0]
        lpart = [list(map(itemgetter(1), g)) for k, g in \
                    groupby(enumerate(ind), lambda x: x[0]-x[1])]
        parts = [[len(lpart[i]),min(lpart[i]),max(lpart[i])] for i in range(len(lpart))]
        inds = [ma_ind[int((p[1]+p[2])/2.0),:] for p in parts] #av. solv. inds

        # concatenate parts of trajectory with identical solvent
        # and stability of > 1 ps
        hshs = [_arr2hash(ind) for ind in inds]; uhshs = list(set(hshs))
        sol = {h:{'traj_inds':np.array([],dtype=int), 'solv_inds':None, 'wts':None} for h in uhshs}
        for i in range(len(hshs)):
            sol[hshs[i]]['solv_inds'] = np.where(inds[i] == True)[0]
            if parts[i][0] >= 1000: # 'stable' intervals > 1ps
                sol[hshs[i]]['traj_inds'] = \
                    np.hstack((sol[hshs[i]]['traj_inds'], lpart[i]))
        # convert to list
        sol = [sol[h] for h in sol if len(sol[h]['traj_inds']) > 0]
        
        # add minor "unstable" trajectory snipets
        # note that this is to exclude 'reactions' from statistics (if wanted)
        for i in range(len(inds)):
            if parts[i][0] < 1000:
                sol.append({'traj_inds':np.array(lpart[i]), \
                            'solv_inds':np.where(inds[i] == True)[0]})
        
        # check for solvent consistency in snippets
        if not np.all([np.array_equal(np.mean(ma_ind[s['traj_inds'],:], \
            axis=0), ma_ind[s['traj_inds'][0],:]) for s in sol]):
            print('Warning: changing water composition for nsolv:%i'%ns)
            print(np.mean(ma_ind[sol[0]['traj_inds'],:],axis=0))
            print(ma_ind[sol[0]['traj_inds'][0],:])
            raise Exception("changing water composition")
        out_sol = out_sol + sol
    
    # check consistent traj length
    assert sum([len(s['traj_inds']) for s in out_sol]) == ma_ind[:,0].size
    return(out_sol)
            

# TODO: make an `ensemlbe` function for h2o adsorption (and others)
def evaluate_h2o_adsorption(md_traj, site_dict, dchs=2.55, ind_subs=[], md_filter="adsorbate_frequency"):
    """
      function to output solvent (h2o) adsorption onto the substrate
      NOTE: this is only written for adsorbate-free trajectories
            updates to be done soon
 
      Parameters
      ----------
      md_traj : mdtraj object
      site_dict : dict (TODO: optional --> call site-fct internally)
        information about adsorption sites as given by get_slab_sites in
        simsoliq.analyze.sampling_sites
      dchs : global distance criterion for water adsorption
      ind_subs : list/np-array (optional)
        array to give substrate indeces
      md_filer : str or list of indeces
        filter for solvent composition: "adsorbate_frequency", "None" or 
        list of indeces which water structure snippets to use
 
      Returns
      -------
      dout : dict
        detailed data of water adsorption
    
    """
    
    # evalute top-atom layer (relevant for water adsorption on steps)
    top_data = get_top_slab_atoms(md_traj, site_dict, ind_subs=ind_subs)
    

    # obtain solvent composition - add information
    sollist = partition_solvent_composition(md_traj)
    sollist = _add_wts_solvent_composition(md_traj, sollist)
    sollist = _add_adsorbate_solvent_composition(md_traj, sollist, ind_subs=ind_subs)


    ############################################################
    # filter by "adsorbate frequency" (make external function) #
    if md_filter == "adsorbate_frequency":
        sol_i = _filter_adsorbate_frequency(md_traj, sollist)
        
    elif md_filter == "None":
        sol_i = list(range(len(sollist)))

    else: # given indece array which solvent-structure snippets to use
        sol_i = md_filter

    # always filter problematic water structures out (have no wts)
    sol_i = [i for i in sol_i if len(sollist[i]['wts']) > 0]
        
    nskip = sum([len(sollist[i]['traj_inds']) for i in range(len(sollist)) \
        if i not in sol_i])
    sollist = [sollist[i] for i in sol_i]

    if nskip > 0:
        print("warning: %i snapshots skipped -- due to filter &/or bad water composition"%nskip)
    ############################################################

    # data-format twisted: site-based output-data
    d = {s:[] for s in range(len(site_dict['tags']))}
    
    ssites = site_dict['coord']; tagme = top_data['itags']
    dsm = 1.0 #dsm is distance of site-position from atom
    a = md_traj.get_traj_atoms()
    cell = a[0].get_cell()
    
    for i in range(0,md_traj.mdtrajlen):
        dd = [sollist[ii] for ii in range(len(sollist)) if i in sollist[ii]['traj_inds']]
        if len(dd) > 0: # only if valid solvent structure
            dd = dd[0]
            spos = a[i].get_scaled_positions()
            sme = spos[top_data['itop'],:]
            # iterate H2O oxygen / solvent center-atom
            for io in dd['wts']:
                # distance to sites (from site_dict)
                dist, mind = __get_distances_order(ssites, spos[io,:], cell) 
                dind = mind[np.where(dist < dchs-dsm)[0]]
                if len(dind) > 0:
                    min_ind = dind[0]
                    d[min_ind].append([i,io])

                # distance to slab top-row atom centers; only enter if site-equivalent 
                # has not already been entered --> enter into site entry
                dist, mind = __get_distances_order(sme, spos[io,:], cell) 
                dind = mind[np.where(dist < dchs)[0]]
                if len(dind) > 0 and (len(d[tagme[dind[0]]]) == 0 or \
                    not np.array_equal(d[tagme[dind[0]]][-1], [i,io])):
                    min_ind = dind[0]
                    d[tagme[min_ind]].append([i,io])
    
    for ste in d:
        d[ste] = np.array(d[ste])
    dout = {'tags':site_dict['tags'], 'tlen':len(a), 'sh2o':d}
    return(dout)


def __get_distances_order(parray, pt, cell): 
    dvec = np.dot(_correct_vec(parray - pt),cell)
    dist = np.linalg.norm(dvec, axis=1)
    mind = dist.argsort(); dist = dist[mind] # only append minimum
    return(dist, mind)


def _add_wts_solvent_composition(md_traj, out_sol):
    """
      helper function to add water structure dict `wts` solvent_composition
      (see function `partition_solvent_composition`). wts is not added
      if no matching solv_inds are found
    
    """
    # add wts which are wind consistent
    for i in range(len(out_sol)):
        sol = out_sol[i]
        winds = sol['solv_inds']
        for t in sol['traj_inds']:
            wts = md_traj._get_solvent_indices(snapshot=t)
            wts_inds = _convert_wts_inds(wts)
            if np.array_equal(np.sort(winds), wts_inds):
                sol.update({'wts':wts})
                break
        if 'wts' not in sol: # problematic water structure
            sol.update({'wts':{}})
    return(out_sol)


def _convert_wts_inds(wts):
    """
      helper function to convert water structure dict `wts` to list
      of water indeces
    
    """
    o = list(wts.keys())
    h = [ww for w in wts.values() for ww in w]
    return(np.sort(h+o))
    

def _add_adsorbate_solvent_composition(md_traj, sollist, ind_subs=[]):
    """
      helper function to add adsorbate indices to solvent_composition
      (see function `partition_solvent_composition`).
      TODO: may also include option to give list (custom filter)
    
    """
    if len(ind_subs) == 0:
        ind_subs = md_traj._get_substrate_indices()
    atoms0 = md_traj.get_single_snapshot(n=0)
    natoms = len(atoms0)

    # iteratre sollist and determine wts, ainds per composition
    for i in range(len(sollist)):
        isolv = sollist[i]['solv_inds']
        ainds = [i for i in range(natoms) if i not in np.hstack((isolv, ind_subs))]
        sollist[i].update({'ads_inds':ainds})

    return(sollist)

    
def _filter_adsorbate_frequency(md_traj, sollist):
    """
      helper function to folter through solvent_composition
      (see function `partition_solvent_composition`) and 
      return only snippets with most frequent adsorbate (other
      snippets may contain different adsorbate due to reactions)
    
    """
    atoms0 = md_traj.get_single_snapshot(n=0)
    sym = atoms0.get_chemical_symbols()
    ads = [''.join(np.sort([sym[ii] for ii in sollist[i]['ads_inds']])) \
        for i in range(len(sollist))]
    tad = [len(sollist[i]['traj_inds']) for i in range(len(sollist))]
    
    # count occurrences of adsorbates
    uads = {a:0 for a in ads}
    for i in range(len(ads)):
        uads[ads[i]] += tad[i]
    vals = list(uads.values())
    c_ads = list(uads.keys())[np.argmax(vals)]

    if len(set(ads)) > 1:
        print("filtered for %s among %s"%(c_ads, str(set(ads))))

    sol_i = [i for i in range(len(sollist)) if ads[i] == c_ads]
    return(sol_i)


def summarize_h2o_adsorption_output(dout, tstart=0):
    """
      function to summarize output from 
      NOTE: this is only written for adsorbate-free trajectories
            updates to be done soon
 
      Parameters
      ----------
      dout : dict
        h2o adsorption data as given by `evaluate_h2o_adsorption`
      tstart : int
        starting point for sampling
    
    """
    # TODO: this needs to go into a summarize function and a general
    #       output function
    # cut data at starting point - remove non-sampled sites
    dads = {s:dout['sh2o'][s][np.where(dout['sh2o'][s][:,0] >= tstart)[0]] \
        for s in dout['sh2o'] if len(dout['sh2o'][s]) > 0}
    tlen = dout['tlen'] - tstart
    print(dout.keys())
    tags = dout['tags']
    
    # counter for site-type
    nsites = {tags[stype]:0 for stype in dads} 
    for s in dads:
        nsites[tags[s]] += len(dads[s]) #number of entries
    nsites.update({'tot':sum([nsites[s] for s in nsites])})

    # average over trajectory length
    for s in nsites:
        nsites[s] /= float(tlen)

    return(nsites)


