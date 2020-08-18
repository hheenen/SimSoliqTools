"""
This module contains functions to analyze the time-propagation
of the solvent (and adsorbate) of an mdtraj object

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
    # (8) make function H2O adsorption
    # (9) make function for adsorbate tracking


    # NOTE: eventual split-up of this module if applicable

def partition_solvent_composition(md_traj): 
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
        # TODO: lpart function probably in helper_function
        lpart = [list(map(itemgetter(1), g)) for k, g in \
                    groupby(enumerate(ind), lambda x: x[0]-x[1])]
        parts = [[len(lpart[i]),min(lpart[i]),max(lpart[i])] for i in range(len(lpart))]
        inds = [ma_ind[int((p[1]+p[2])/2.0),:] for p in parts] #av. solv. inds

        # NOTE: not sure if this separation is really necessary
        # concatenate parts of trajectory with identical solvent
        # and stability of > 1 ps
        hshs = [_arr2hash(ind) for ind in inds]; uhshs = list(set(hshs))
        sol = {h:{'traj_inds':np.array([],dtype=int), 'solv_inds':None} for h in uhshs}
        for i in range(len(hshs)):
            sol[hshs[i]]['solv_inds'] = np.where(inds[i] == True)[0]
            if parts[i][0] >= 1000: # 'stable' intervals > 1ps
                sol[hshs[i]]['traj_inds'] = \
                    np.hstack((sol[hshs[i]]['traj_inds'], lpart[i]))
        # convert to list
        sol = [sol[h] for h in sol if len(sol[h]['traj_inds']) > 0]
        
        # add minor "unstable" trajectory snipets
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
def evaluate_h2o_adsorption(md_traj, site_dict, dchs=2.55):
    """
      function to output solvent (h2o) adsorption onto the substrate
      NOTE: this is only written for adsorbate-free trajectories
            updates to be done soon
 
      Parameters
      ----------
      md_traj : mdtraj object
      site_dict : dict (TODO: optional)
        information about adsorption sites as given by get_slab_sites in
        simsoliq.analyze.sampling_sites
      dchs : global distance criterion for water adsorption
 
      Returns
      -------
      dout : dict
        detailed data of water adsorption
    
    """
    
    # evalute top-atom layer (relevant for water adsorption on steps)
    top_data = get_top_slab_atoms(md_traj, site_dict)
    
    # TODO: can use mdtraj functions here? NOTE: skipped for now so H2O adsorption for clean surfaces can be used
    # pre-determine indeces for water -- can use mdtraj functions???? 
  # ainds0, winds0 = _get_water_adsorbate_inds(f)
  # winds0 = [i for i in winds0 if a[0].get_chemical_symbols()[i] in ['H','O']] # remove Na+ from water
  # dat = eval_sys_inds(a, ainds0, winds0)

    # NOTE: for now use mdtraj based water identify --> to make hash based list (assuming no adsorbate)
    sollist = partition_solvent_composition(md_traj)
    if len(sollist) > 1:
        raise Exception('changing water composition! --> likely adsorbate present Not finally implemented')
    wts = md_traj._get_solvent_indices(snapshot=sollist[0]['traj_inds'][0])
    dat = {_arr2hash(sollist[0]['solv_inds']):{'winds':sollist[0]['solv_inds'], 'ainds':[], 'tinds':sollist[0]['traj_inds'], 'wts':wts}}

    # data-format twisted: site-based
    d = {s:[] for s in range(len(site_dict['tags']))}
    
    ssites = site_dict['coord']; tagme = top_data['itags']
    count = 0; dsm = 1.0 #dsm is distance of site-position from atom
    a = md_traj.get_traj_atoms()
    cell = a[0].get_cell()
    
    for i in range(0,md_traj.mdtrajlen):
        # choose right adsorbate configuration
        dd = [dat[hsh] for hsh in dat if i in dat[hsh]['tinds']]
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
    
    # counter for site-type
    nsites = {stype:0 for stype in dads} 
    for s in dads:
        nsites[s] += len(dads[s]) #number of entries
    nsites.update({'tot':sum([nsites[s] for s in nsites])})

    # average over trajectory length
    for s in nsites:
        nsites[s] /= tlen

    return(nsites)





# TODO: incorporate later!!
#########################################################################################
#########################################################################################
#########################################################################################
def _get_water_adsorbate_inds(f):
    sdat = _load_pickle_file('%s/site_sampling.pkl'%f)
    if np.all([f.find(m) == -1 for m in ['OCCHO','CHO','COH','CO','OH','OOH','Li+','K+','Na+']]):
        ainds = []; winds = np.where(sdat[0]['atomic_numbers'] != symbls.index(f[:2]))[0]
    else:
        rads = f.split('_')[-2]; success = False
        for i in range(len(sdat)):
            ainds = sdat[i]['ads_inds']
            winds = sdat[i]['water_inds']
            atno = sdat[i]['atomic_numbers']
            aord = ''.join(np.sort([symbls[atno[ii]] for ii in ainds]))
            if aord == ''.join(np.sort([s for s in rads])).strip('+'):
                success = True
                break
        if not success:
            if f in backup_ainds:
                ainds = backup_ainds[f]
                winds = np.array([i for i in range(atno.size) \
                    if i not in ainds and atno[i] != symbls.index(f[:2])])
            else:
                raise Exception('could not determine ainds for %s'%f)
    # weird exception - not found before (doesn't matter for msd)
   #if f == 'Cu211_15H2O_OCCHO_05':

    return(ainds, winds)

def eval_sys_inds(a, ainds0, winds0):
    # set-up starting inds
    ainds0 = np.array(ainds0); winds0 = np.array(winds0)
    ainds = ainds0; winds = winds0; allinds = np.array(winds.tolist()+ainds.tolist())
 
    for i in range(len(a)): # find 1st wts!
        wts = identify_water(a[i])
        fwts = np.sort([w for w in wts] + [wi for o in wts for wi in wts[o]])
        if np.array_equal(winds,fwts):
            break
    
    nskip = 0; nchange = 0
    dat = {_make_hash(winds0):{'winds':winds0, 'ainds':ainds0, 'tinds':[], 'wts':wts}}
    for i in range(len(a)):
        wts = identify_water(a[i]) # may sort same H to different Os
        fwts = np.unique([w for w in wts] + [wi for o in wts for wi in wts[o]])
        # handle special cases
        if fwts.size != winds.size:
            nskip += 1
            continue # skip any non-identifyable
        else:
            if not np.array_equal(winds,fwts):
                winds = fwts
                # need to determine new ainds
                ainds = np.array([ii for ii in allinds if ii not in winds])
                nchange += 1
                #print('winds changed in step %i to %s'%(i,str(ainds)))
        # make entry:
        hsh = _make_hash(winds)
        if hsh not in dat:
            dat.update({hsh:{'winds':winds, 'ainds':ainds, 'wts':wts, 'tinds':[]}})
        dat[hsh]['tinds'].append(i)
    print('skipped %i images / adsorbates changed %i times'%(nskip,nchange))
    return(dat)

#########################################################################################
#########################################################################################
#########################################################################################
