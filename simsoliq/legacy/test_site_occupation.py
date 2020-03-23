#!/usr/bin/env python

import sys, os
import numpy as np
sys.path.append("/home/cat/heenen/Workspace/benchmarking_calcs/Cu_AIMD_runs/scripts_vasp")
from post_process_traj import _read_restart_atoms
from transform_lattice_constant_standalone import identify_water

sys.path.append('/home/cat/heenen/Workspace/TOOLS/ASE-DTU/ase')
from ase.io import read, write
from ase.visualize import view
from ase.atoms import Atoms
from ase.build import fcc111
from ase.data import chemical_symbols
from copy import deepcopy

sys.path.append("/home/cat/heenen/Workspace/test_QE/surface_sampling/site_enumeration")
from test_site_enumeration import get_Cu_slab_adsorption_sites

from cathelpers.atom_manipulators import _correct_vec
from cathelpers.misc import _load_pickle_file, _write_pickle_file
from time import time

from operator import itemgetter
from itertools import *

import hashlib
import base64

def get_sample_sites(tatoms,surface,repeat):
    # get fixed atom type
    try:
        type_fix = np.unique(tatoms.get_atomic_numbers()[tatoms.constraints[0].index])[0]
    except IndexError:
        type_fix = np.unique(tatoms.get_atomic_numbers())[-1]
        print('missing constraints in traj-file - assuming heaviest element for slab')

    # make surface sites
    lats = {'Cu':3.56878996, 'Pt': 3.934289636, 'Au': 4.167052576}
    # only ASE-oriented cells
    site_dict, vec_dict, psite_dict = get_Cu_slab_adsorption_sites(\
        surface,layer=9,vacuum=10,cell_shift=[0.,0.],lat=lats[chemical_symbols[type_fix]])
    
    shift_cell = tatoms.get_cell();
    if np.array_equal(np.sort(surface),[1,1,2]):
        shift_cell[0,0] /= repeat[0]/3. # x - step length
        shift_cell[1,1] /= repeat[1] # y
    elif np.array_equal(surface,[1,1,1]):
        c = fcc111(chemical_symbols[type_fix], a=lats[chemical_symbols[type_fix]], \
                                    size=(1,1,4), vacuum=10.0, orthogonal=False)
        shift_cell = c.get_cell()
    else:
        raise Exception('cell %s not implemented'%str(surface))
    psite_dict = _extendsites(site_dict,shift_cell)
    
    ##################################################
    # stupid hack - used wrong functionality of CatKit
    if np.array_equal(np.sort(surface),[1,1,2]):
        correction = {'Cu':2.524/2.0}
        for site in ['bridge-a','bridge-c']:
            nsites = deepcopy(psite_dict[site]); nsites[:,1] += correction[chemical_symbols[type_fix]]
            psite_dict[site] = np.vstack((psite_dict[site],nsites))
    if np.array_equal(surface,[1,1,1]):
        correction = {'Cu':[[-0.64,1.0925],[0.64,1.0925]],\
                      'Au':[[-0.737,1.276],[1.473,0.0]],\
                      'Pt':[[-0.6955,1.2045],[0.6955, 1.2045]]}
        ncorr = []
        for corr in correction[chemical_symbols[type_fix]]:
            nsites = deepcopy(psite_dict['bridge'])
            nsites[:,0] += corr[0]; nsites[:,1] += corr[1]
            ncorr.append(nsites)
        psite_dict['bridge'] = np.vstack(tuple([psite_dict['bridge']]+ncorr))
    ##################################################
    
    # filter sites
    psite_dict = {p:_filter_sites_cell(psite_dict[p],tatoms.get_cell()) \
                                                for p in psite_dict}
    # prep sites
    stags = []; sites = np.array([])
    for site in psite_dict:
        stags += [site]*psite_dict[site][:,0].size
        if sites.size == 0:
            sites = psite_dict[site]
        else:
            sites = np.vstack((sites,psite_dict[site]))
    zmax = sites[:,2].max()
    
    ind_slab = np.where(tatoms.get_atomic_numbers() == type_fix)[0]
    vz = tatoms.positions[ind_slab,2].max() - zmax + 1.0
    sites[:,2] += vz
    
    scaled_sites = np.linalg.solve(tatoms.get_cell(complete=True).T,
                                     sites.T).T

   ## to visualize
   #sa = Atoms(numbers=np.ones(sites[:,0].size)*2.,positions=sites,cell=tatoms.get_cell())
   #view(tatoms+sa)
   ##write("tocheck_atoms.traj",tatoms+sa)
   #assert False
    return({'coord':sites,'scaled_coord':scaled_sites,'tags':np.array(stags)})

def _extendsites(sitedict,cell): #copied from site-enumeration
    shifts = [[a,b] for a in [-3,-2,-1,0,1,2,3,4,5,6] for b in [-3,-2,-1,0,1,2,3,4,5,6]]
    psitedict = {sk:[] for sk in sitedict}
    for shift in shifts:
        for sk in sitedict:
            pos = deepcopy(sitedict[sk])
            pos[:2] += np.dot(shift,cell[:2,:2])
            psitedict[sk].append(pos)
    psitedict = {sk:np.array(psitedict[sk]) for sk in psitedict}
    return(psitedict)
    
def _filter_sites_cell(sites,cell):
    lxy = np.array([cell[0,0],cell[1,1]])
    nind = [i for i in range(sites[:,0].size) if not \
        np.any((sites[i,:2] < 0).tolist()+(sites[i,:2] - lxy >= 0).tolist())]
    return(sites[nind,:])
    
def __get_ads_ind(atoms, exclude=[]):
    try:
        fixt = np.unique(atoms.get_atomic_numbers()[atoms.constraints[0].index])
    except IndexError:
        fixt = [np.unique(atoms.get_atomic_numbers())[-1]]
        print('missing constraints in traj-file - assuming heaviest element for slab')
    indslab = np.vstack(tuple([np.where(atoms.get_atomic_numbers() == f)[0] \
        for f in fixt]))[0]
    return([a for a in range(len(atoms)) if a not in np.hstack((indslab,exclude))])

def sample_surface_interaction(ssites,atoms,dcut=1.5):
    # isolate water and adsorbate inds
    wlist = sample_solvent(atoms) # dict of averaged water-inds
    out = []
    for wentry in wlist:
        winds = wentry['solv_inds']
        tatoms = [atoms[i] for i in wentry['traj_inds']]
        adict, wdict, ainds = _get_distance_trajectory(tatoms, winds, ssites, dcut)
        aintdist = _get_internal_adsorbate_distances(tatoms, ainds)
        out.append({'adsorbate':adict, 'solvent':wdict, \
                    'ads_inds':ainds, 'water_inds':winds, \
                    'ssites':ssites, 'traj_inds':wentry['traj_inds'],\
                    'dist_ads_internal':aintdist,\
                    'atomic_numbers':tatoms[0].get_atomic_numbers()})
    return(out)
        
def _get_internal_adsorbate_distances(tatoms, ainds):
    # compute internal distance - to monitor all stability of adsorbate
    cell = tatoms[0].get_cell()
    ##tinds = {ai:[ind for ind in ainds if ind != ai] for ai in ainds}
    ##dists = {ai:np.zeros((len(ainds)-1,len(tatoms))) for ai in ainds}
    # cmbi -- nested loop for internal adsorbate combinations
    cmbi = [[ainds[i],ainds[j]] for i in range(0,len(ainds)-1) for j in range(i+1,len(ainds))]
    dists = np.zeros((len(cmbi),len(tatoms)))

    for t in range(len(tatoms)):
        rpos = tatoms[t].get_scaled_positions()
        for i in range(len(cmbi)):
            ai = cmbi[i]
            dvec = _correct_vec(rpos[ai[0],:] - rpos[ai[1],:])
            dvec = np.dot(dvec.T,cell)
            d = np.linalg.norm(dvec)#,axis=1)
            dists[i,t] = d
    return({'intd':dists, 'cmbi':cmbi})

def _get_distance_trajectory(atoms, winds, ssites, dcut):
    ainds = __get_ads_ind(atoms[0],winds)

    pos_list = [a.get_scaled_positions() for a in atoms]
    cell = atoms[0].get_cell()
    # screen adsorbate binding
    
    # iterate each atom to find which ones are binding:
    adict = _filter_distances(ainds, pos_list, cell, ssites, dcut)
    wdict = _filter_distances(winds, pos_list, cell, ssites, dcut)
    return(adict, wdict, ainds)

def __fgauss(d,c):
    #f = lambda d,c: np.exp(-(d**2.0)/(2*c**2))
    return(np.exp(-(d**2.0)/(2*c**2)))

def __get_gauss_weights(ssites):
    pos = ssites['coord']; spos = ssites['scaled_coord']; tags = ssites['tags']
    
    # get shortest distances to neighboring sites (half distance == 50%)
    mind = {tag:99. for tag in set(tags)}
    for i in range(pos[:,0].size):
        #dvec = _correct_vec(spos - spos[i,:])
        dvec = pos - pos[i,:]
        d = np.linalg.norm(dvec,axis=1)
        d = d[np.where(d != 0.0)[0]]
        mind[tags[i]] = min(mind[tags[i]],d.min())
    
    # adjust values to overall min to avoid over weighting of top-sites
    mind = {k:min(mind.values()) for k in mind}

    # create according Gaussian weights - c = standard sigma
    # function shall fall below 0.1% at other site, alternatively 50% at d/2 but 5% at other site
    gw = {}
    for k in mind:
        d = mind[k]; c = 10.
        while __fgauss(d,c) > 0.001: 
            c -= 0.005 
        gw.update({k:deepcopy(c)})
    return(gw)

def analyze_adsorbate_site_traj(awdict,batom):
    # returns instantaneous closest (min) and distance weighted traj
    k = 'adsorbate'
    gwts = __get_gauss_weights(awdict['ssites']) #gaussian site weights
    if len(awdict[k]) > 0:
        d_batoms = {}; site_traj = {}; site_wts = {}; tot_dmax = {}
        #iterating adsorbate atoms
        for a in awdict[k]: 
            d = awdict[k][a]['distances']
            stags = awdict[k][a]['stags']
            ttraj = awdict['traj_inds']

            # site-weight
            dsh = np.array([__fgauss(d[ii,:],gwts['-'.join(stags[ii].split('-')[:-1])]) for ii in range(d[:,0].size)])
            nfdsh = dsh.sum(axis=0)
            ndsh = dsh/nfdsh
            ndsh = {stags[s]:ndsh[s,:] for s in range(len(stags))}

            # for 'min-distance' selection
            ind_c = d.argmin(axis=0) # closest site at timestep
            occ = {stags[s]:ttraj[np.where(ind_c == s)[0]] for s in range(len(stags))} #new (correct?)
            occ = {k:occ[k].astype(int) for k in occ} # indices! convert for later use
            
            # d_batoms collects min distances for each "bound atoms"
            # for choice of actual "bound atoms"
            if awdict['atomic_numbers'][a] == batom:
                d_batoms.update({a:np.mean(d.min(axis=0))})
                site_traj.update({a:occ})
                site_wts.update({a:ndsh})
                # max distance per timestep - track desorption
                tot_dmax.update({a:d.min(axis=0)})
        
        # choose binding atom
        db = list(d_batoms.items()); bi = [b[0] for b in db]; bd = [b[1] for b in db]
        choice = bi[np.argmin(bd)]
        
        return(site_traj[choice], site_wts[choice], tot_dmax[choice])
 

def _filter_distances(ainds, pos_list, cell, ssites, dcut):
    #   save each atom with dmin < dcut; and according d-arrays in dict
    ddict = {}
    apos = {ai:np.array([p[ai,:] for p in pos_list]) for ai in ainds}
    for ai in apos:
        ds = np.zeros((ssites['scaled_coord'][:,0].size, apos[ai][:,0].size))
        for t in range(apos[ai][:,0].size): # times series of distances
            dvec = _correct_vec(ssites['scaled_coord']-apos[ai][t,:])
            dvec = np.dot(dvec,cell)
            d = np.linalg.norm(dvec,axis=1)
            ds[:,t] = d
        # select if atom has relevant distance
        if ds.min(axis=1).min() < dcut: # magic number!?
            # close-distance sites 
            mininds = np.array([np.any(ds[i,:] < dcut) for i in range(ds[:,0].size)]).nonzero()[0].astype('int')
            ddict.update({ai:{'stags':[ssites['tags'][i]+'-%i'%i for i in mininds],\
                'distances':ds[mininds,:]}})
    return(ddict)

def sample_solvent(atoms):
    ''' (1) identify_water for each snapshot
        (2) take mean/dominant assignment from 10 surrounding snaphots
            Exceptions for deviating assignments - possible future issue
            --> return one list of inds if always same
            --> return time-limits if it is not
    '''
    # read all indices
    ma_ind = np.zeros((len(atoms),len(atoms[0])))#,dtype=bool)
    for i in range(len(atoms)):
        idw = identify_water(atoms[i])
        indw = [io for io in idw] + \
            np.array([ih for io in idw for ih in idw[io]]).flatten().tolist()
        ma_ind[i,indw] = 1.0 #True
    
    # just fyi - check if same number of waters
    if not len(np.unique(ma_ind.sum(axis=1))) == 1:
        print('Warning: non-constant identification of water')
    
    # averaging water
    av_ma_ind = np.zeros(ma_ind.shape)
    da = 20 # should only show averages dominant > 40 fs (change to 100fs?)
    for i in range(ma_ind[:,0].size): # here could be problems in OH-traj
        av = ma_ind[max(0,i-da):min(i+da,ma_ind[:,0].size),:].mean(axis=0)
        av = np.around(av,0)
        av_ma_ind[i,:] = av
    ma_ind = av_ma_ind
    
    # sample snippets of constant solvent composition
    nsolv, nfreq = np.unique(ma_ind.sum(axis=1),return_counts=True)
    out_sol = []
    for ns in nsolv: #iterate changing solvent composition
        ind = np.where(ma_ind.sum(axis=1) == ns)[0]
        lpart = [list(map(itemgetter(1), g)) for k, g in \
                    groupby(enumerate(ind), lambda x: x[0]-x[1])]
        parts = [[len(lpart[i]),min(lpart[i]),max(lpart[i])] for i in range(len(lpart))]
        inds = [ma_ind[int((p[1]+p[2])/2.0),:] for p in parts] #av. solv. inds
        
        # concatenate parts of trajectory with identical solvend
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

def _arr2hash(arr):
    arr[arr == 0] = 0
    hashed = hashlib.md5(arr).digest()
    hashed = base64.urlsafe_b64encode(hashed)
    return(hashed)

def _group_site_inds(sinds):
    sinds_p = {}
    for site in sinds:
        ind = sinds[site]
        lpart = [list(map(itemgetter(1), g)) for k, g in \
                groupby(enumerate(ind), lambda x: x[0]-x[1])]
        sinds_p.update({site:lpart})
    return(sinds_p)


def _post_surface_sampling(f):
    surface = tuple([int(e) for e in f.split('/')[-1].split('_')[0][2:5]])
    atoms = _read_restart_atoms(f)
    ssites = get_sample_sites(atoms[0],surface=surface,repeat=(3,3))
    site_traj_data = sample_surface_interaction(ssites,atoms)
    return(site_traj_data)

def _get_sample_sites_atoms(f):
    # solely for testing
    surface = tuple([int(e) for e in f.split('/')[-1].split('_')[0][2:5]])
    atoms = read(f+'/POSCAR')  #_read_restart_atoms(f)
    ssites = get_sample_sites(atoms,surface=surface,repeat=(3,3)); sites = ssites['coord']
    # to visualize
    sa = Atoms(numbers=np.ones(sites[:,0].size)*2.,positions=sites,cell=atoms.get_cell())
    return(atoms+sa)

def __get_adstag_fname(fname):
    atag = fname.split('_')[2]
    if np.all([a in fname.split('_') for a in ['OCCHO','Na+']]):
        atag = 'NaOCCHO'
    return(atag)

def get_ads_site_traj(f, reduced=True, gaussian_weighted=False):
    fpkl = '%s/site_sampling.pkl'%f; atag = f.split('/')[-1].split('_')[2]
    stag = f.split('/')[-1].split('_')[0]
    atag = __get_adstag_fname(f.split('/')[-1])
    if os.path.isfile(fpkl):
        awdict = _load_pickle_file(fpkl)
        straj = _get_ads_site_traj(awdict, stag, atag, reduced=reduced, gaussian_weighted=gaussian_weighted)
    return(straj)
                
def __split_chem_str(cstr):
    # could be overloaded split method
    indu = [i for i in range(len(cstr)) if cstr[i].isupper()] + [len(cstr)]
    splt = [cstr[indu[i]:indu[i+1]] for i in range(len(indu)-1)]
    return(splt)

def _get_ads_site_traj(awdict, stag, atag, reduced=True, gaussian_weighted=False):
    bound_atoms = {k:6 for k in ['CHO','CO','COH','OCCHO','NaOCCHO']}
    bound_atoms.update({k:8 for k in ['OH','OOH']})
    deslim = {'111':2.0, '211':2.3}
    tsplt = ''.join(np.sort(__split_chem_str(atag)))

    straj = {}; wtraj = {}; wtct = 0
    # this is changed to total length vector to be compatible with energy vectors!
    # length of counted trajectory for temp sites in weighted trajectory #TODO: delete if not necessary anymore
  ##countlen = sum([prt_a['traj_inds'].size for prt_a in awdict \
  ##    if (len(prt_a['traj_inds']) > 1000 and len(prt_a['ads_inds']) > 0 \
  ##    and ''.join(np.sort([chemical_symbols[prt_a['atomic_numbers'][ii]] \
  ##                            for ii in prt_a['ads_inds']])) == tsplt)])
    totlen = sum([len(prt_a['traj_inds']) for prt_a in awdict]); nan_inds = np.array([])
    if atag in bound_atoms:
        if 'dist_ads_internal' in awdict[0]:
            # choose initial coordinates from stable struc AND ads presence
            ad_int0 = [awdict[w]['dist_ads_internal']['intd'][:,0] \
                for w in range(len(awdict)) if len(awdict[w]['traj_inds']) > 1000]
            ad_int0 = [a for a in ad_int0 if np.any([len(a) == \
                sum(range(len(atag.strip(ion)))) for ion in ['Na','Li','K']])][0]
        #go through snippets
        for prt_a in awdict:
            # only consider larger samples of a state and with adsorbate
            if len(prt_a['traj_inds']) > 1000 and \
                len(prt_a['ads_inds']) > 0:
                st, stwt, dmax = analyze_adsorbate_site_traj(prt_a,\
                                                    batom=bound_atoms[atag])
                
                # right chemical composition?
                atno = prt_a['atomic_numbers']; iads = prt_a['ads_inds']
                aord = ''.join(np.sort([chemical_symbols[atno[ii]] for ii in iads]))
                #if aord != ''.join(np.sort([c for c in atag])): # no? -> add to odd stuff 'chmst'
                if aord != tsplt: # no? -> add to odd stuff site 'chmst'
                    st = {'chmst':np.hstack(tuple([st[ste] for ste in st]))}
                
                # check for desorption
                des = np.where(dmax > deslim[stag[2:5]])[0]

                dis = np.array([])
                # check for surface dissociation of adsorbate:
                if 'dist_ads_internal' in prt_a:
                    ad_int = prt_a['dist_ads_internal']['intd']; dis = np.array([])
                    # get difference to initial internal distances
                    for i in range(ad_int[:,0].size):
                        #print(i, prt_a['dist_ads_internal']['cmbi'][i])
                        #dref = np.absolute(ad_int[i,:] - ad_int[i,0])
                        dref = np.absolute(ad_int[i,:] - ad_int0[i])
                        ind = np.where(dref > 1.5)[0]
                        dis = np.hstack((dis,ind))

                # diss- and desorpt.. --> count to nan (wtraj); remove in straj
                indd = np.hstack((des, dis)).astype(int)
                if indd.size > 0:
                    nan_inds = np.hstack((nan_inds,prt_a['traj_inds'][indd]))
                    st = {ste:[i for i in st[ste] if i not in indd] for ste in st}
                    
                # work into existing dict
                for ste in st:
                    if ste not in straj:
                        straj.update({ste:np.array([])}); wtraj.update({ste:np.zeros(totlen)}) ##countlen)})
                    straj[ste] = np.hstack((straj[ste],st[ste]))
                    if ste != 'chmst':
                        #wtraj[ste] = np.hstack((wtraj[ste],stwt[ste]))
                       ##wtraj[ste][wtct:wtct+prt_a['traj_inds'].size] = stwt[ste]
                        wtraj[ste][prt_a['traj_inds']] = stwt[ste]
                    else: # for post_correction of chmst entries to total wtraj
                        nan_inds = np.hstack((nan_inds,prt_a['traj_inds']))
              ##if ste != 'chmst':
              ##    wtct += prt_a['traj_inds'].size
            else:
                nan_inds = np.hstack((nan_inds,prt_a['traj_inds']))
        # special cases -- find 'manual-outsort'
        if np.any(['manual-outsort' in prt_a for prt_a in awdict]):
            manout = np.unique([prt_a['manual-outsort'] for prt_a in awdict if 'manual-outsort' in prt_a])
            nan_inds = np.unique(np.hstack((nan_inds, manout)))
            straj = {ste:np.array([i for i in st[ste] if i not in manout]) for ste in straj}
        for sste in wtraj: #overwrite unsortable sites to np.nan
            wtraj[sste][nan_inds.astype(int)] = np.nan
        
        # error check all weighted trajs same length
        if 'chmst' in wtraj:
            del wtraj['chmst']
        if np.unique([wtraj[s].size for s in wtraj]).size > 1: #0==ONLY chmst
            raise Exception("weighted trajectories are NOT all the same length")

        if reduced:
            straj = _reduce_sites_inds(straj)
            if gaussian_weighted:
                wtraj = _reduce_sites_weights(wtraj)
        if gaussian_weighted:
            straj = wtraj
    return(straj)

def _reduce_sites_inds(straj):
    # (1) summarizes individual site types 
    # (2) filters out sites represented < 500 fs
    symmkeys = list(set(['-'.join(s.split('-')[:-1]) for s in straj.keys()]))
    rs = {s:np.array([]) for s in symmkeys}
    for s in rs:
        for ss in straj.keys():
            if '-'.join(ss.split('-')[:-1]) == s:
                if straj[ss].size > 0:
                    rs[s] = np.hstack((rs[s],straj[ss]))
    [rs[s].sort() for s in rs]
    rs = {s:rs[s] for s in rs if rs[s].size > 500}
    return(rs)

def _reduce_sites_weights(wtraj):
    # (1) summarizes individual site types 
    symmkeys = list(set(['-'.join(s.split('-')[:-1]) for s in wtraj.keys()]))
    lene = wtraj[list(wtraj.keys())[0]].size
    rs = {s:np.zeros(lene) for s in symmkeys}
    for s in rs:
        for ss in wtraj.keys():
            if '-'.join(ss.split('-')[:-1]) == s:
                if wtraj[ss].size > 0:
                    rs[s] += wtraj[ss]
    # (2) filters out sites represented < 0.005 share
    rs = {s:rs[s] for s in rs if rs[s][np.isnan(rs[s]) == False].sum() > 0.005}
    return(rs)

def __reduce_waters(chem):
    # contract chemical symbols to n(H2O)x
    c = [ch for ch in chem]
    el, count = np.unique(c,return_counts=True)
    if len(el) != 2:
        raise Exception('something to adjust for solvent elements %s'%str(el))
    nH = count[np.where(el == 'H')[0]]; nO = count[np.where(el == 'O')[0]]
    nH2O = nO; nR = nH - 2*nO; sH2O = "%i(H2O)"%nH2O; sR = "%iH"%(nR)
    if nR != 0:
        sH2O += sR
    return(sH2O)
        
def _traj_overview(identifier,site_dat):
    txt_out = 80*'#'+'\n'+20*'#'+'{:^40}'.format(identifier)+20*'#'+'\n'+80*'#'+'\n\n'
    txt_out += "#### %i solvent configurations\n"%(len(site_dat))
    
    # solvent composition
    for i in range(len(site_dat)):
        d = site_dat[i]
        ads = ''.join(np.sort([chemical_symbols[d['atomic_numbers'][j]] \
            for j in d['ads_inds']]))
        water = __reduce_waters([chemical_symbols[d['atomic_numbers'][j]] \
            for j in d['water_inds']])
        try:
            txt_out += '  '+"%i: %i fs, ads=%s, water=%s\n"%(i+1,d['traj_inds'].size,ads,water)
        except AttributeError: # short snippets saved in the past as lists
            txt_out += '  '+"%i: %i fs, ads=%s, water=%s\n"%(i+1,len(d['traj_inds']),ads,water)

    stag = f.split('/')[-1].split('_')[0]
    atag = __get_adstag_fname(f.split('/')[-1])
    ttot = sum([len(d['traj_inds']) for d in site_dat])
    # desorption events:
    d_evnts, diss_e = __read_desorption_dissociation_events(site_dat, stag, atag)
    n_des = sum([dev[1]-dev[0] for dev in d_evnts])
    if len(d_evnts) > 0:
        txt_out += '\n#### desorption events: %i/%i\n    '%(n_des, ttot)
        txt_out += ', '.join(['%i-%i'%tuple(dev) for dev in d_evnts]) + '\n'
    n_dis = sum([ds[1]-ds[0] for ds in diss_e])
    if len(diss_e) > 0:
        txt_out += '\n#### dissociation events: %i/%i\n    '%(n_dis, ttot)
        txt_out += ', '.join(['%i-%i'%tuple(dis) for dis in diss_e]) + '\n'
    
    # visited sites
    straj = _get_ads_site_traj(site_dat, stag, atag, reduced=False, gaussian_weighted=False)
    wtraj = _get_ads_site_traj(site_dat, stag, atag, reduced=False, gaussian_weighted=True)
    wtraj = deepcopy({s:wtraj[s][np.isnan(wtraj[s]) == False] for s in wtraj})
    scount = sum([straj[s].size for s in straj])
    txt_out += '\n'+'#### visited sites (fs/%) | {0:d}/{1:d}:\n'.format(scount,ttot)
    # formating sites
    sstrings = ['{:>12} ({:5d} / {:4.1f})'.format(s,straj[s].size,straj[s].size/ttot*100.) for s in straj if straj[s].size > 0]
    slines = ''.join(['   '+';   '.join(sstrings[j:j+3])+'\n' \
        for j in range(0,len(sstrings),3)])
    #slines += '  %s\n'%'; '.join(sstrings[-(len(sstrings)%3):])
    txt_out += slines
    try:
        wtot = [wtraj[s].size for s in wtraj][0]
    except:
        wtot = 0
    txt_out += '\n'+'---- gaussian weighted sites (%) | {0:d}/{1:d}:\n'.format(wtot,ttot)
    # formating sites
    sstrings = ['{:>12} ({:4.1f})'.format(s,wtraj[s].sum()/wtraj[s].size*100.) for s in wtraj if wtraj[s].sum()/wtraj[s].size > 0.005]
    slines = ''.join(['   '+';   '.join(sstrings[j:j+3])+'\n' \
        for j in range(0,len(sstrings),3)])
    txt_out += slines
    return(txt_out)


def __read_desorption_dissociation_events(awdict, stag, atag):
    bound_atoms = {k:6 for k in ['CHO','CO','COH','OCCHO','NaOCCHO']}; bound_atoms.update({k:8 for k in ['OH','OOH']})
    ddiss = {f:1.5 for f in ['OH','CO','CHO','COH','OCCHO']}; ddiss.update({'OOH':1.5})
    deslim = {'111':2.0, '211':2.3}; dev = []; diss = []
    if atag in bound_atoms:
        if 'dist_ads_internal' in awdict[0]:
            # choose initial coordinates from stable struc AND ads presence
            ad_int0 = [awdict[w]['dist_ads_internal']['intd'][:,0] \
                for w in range(len(awdict)) if len(awdict[w]['traj_inds']) > 1000]
            ad_int0 = [a for a in ad_int0 if np.any([len(a) == \
                sum(range(len(atag.strip(ion)))) for ion in ['Na','Li','K']])][0]
        for prt_a in awdict:
            if len(prt_a['traj_inds']) > 1000 and len(prt_a['ads_inds']) > 0:
                st, stwt, dmax = analyze_adsorbate_site_traj(prt_a,\
                                                    batom=bound_atoms[atag])
                des = prt_a['traj_inds'][np.where(dmax > deslim[stag[2:5]])[0]]
                lpart = [list(map(itemgetter(1), g)) for k, g in \
                    groupby(enumerate(des), lambda x: x[0]-x[1])]
                [dev.append([lp[0],lp[-1]]) for lp in lpart]
                
                if 'dist_ads_internal' in prt_a:
                    ad_int = prt_a['dist_ads_internal']['intd']; dis = np.array([])
                    # get difference to initial internal distances 
                    # if dissociation event changes water structure -- this is not captured
                    # need to take very first water structure
                    for i in range(ad_int[:,0].size):
                        #print(i, prt_a['dist_ads_internal']['cmbi'][i])
                        dref = np.absolute(ad_int[i,:] - ad_int0[i])
                        #dref = np.absolute(ad_int[i,:] - ad_int[i,0])
                        ind = prt_a['traj_inds'][np.where(dref > ddiss[atag])[0]]
                        dis = np.hstack((dis,ind))
                    dis = np.unique(dis)
                    lpart = [list(map(itemgetter(1), g)) for k, g in \
                        groupby(enumerate(dis), lambda x: x[0]-x[1])]
                    [diss.append([lp[0],lp[-1]]) for lp in lpart]
                else:
                    diss = []
    return(dev, diss)

def _detailed_traj_output(identifier,site_dat):
    # isolate detailed trajectory
    #atag = identifier.split('_')[2]
    stag = f.split('/')[-1].split('_')[0]
    atag = __get_adstag_fname(f.split('/')[-1])
    traj_info =  _enumerate_site_trajectory(site_dat, stag, atag)

    # formating trajectory
    txt_out = 80*'#'+'\n'+20*'#'+'{:^40}'.format(identifier)+20*'#'+'\n'+80*'#'+'\n\n'
    trstr = ["%s (%i fs)"%(entry[2],entry[1]-entry[0]) for entry in traj_info]
    trlines = ''.join([' '+' --> '.join(trstr[j:j+4])+'\n' \
        for j in range(0,len(trstr),4)])
    return(txt_out+trlines)

def _enumerate_site_trajectory(site_dat, stag, atag):
    straj = _get_ads_site_traj(site_dat, stag, atag, reduced=False, gaussian_weighted=False)
    straj = {k:straj[k].astype(int) for k in straj} # convert for use as indeces
    traj_info = []
    for key in straj:
        lpart = [list(map(itemgetter(1), g)) for k, g in \
                groupby(enumerate(straj[key]), lambda x: x[0]-x[1])]
        traj_info += [[lp[0],lp[-1],key] for lp in lpart]
        #fstraj[straj[key]] = j; j += 1
    ind = np.argsort([l[0] for l in traj_info])
    traj_info = [traj_info[i] for i in ind]
    return(traj_info)

def evaluate_residence_time(f, tlim=150): 
    site_dat = _load_pickle_file(f+'/site_sampling.pkl')
    stag = f.split('/')[-1].split('_')[0]
    atag = __get_adstag_fname(f.split('/')[-1])
    tdat =  _enumerate_site_trajectory(site_dat, stag, atag)
    # skim site-traj by sites visited < tlim
    # split to previous and following site
    remove = []
    for i in range(1,len(tdat)-1):
        dt = tdat[i][1] - tdat[i][0]
        if dt < tlim:
            # prev none 'None' entry
            for pi in list(range(0,i))[::-1]:
                if tdat[pi][0] != None:
                    break
            tdat[pi][1] += dt//2
            tdat[i+1][0] -= dt//2
            tdat[i][0] = tdat[i][1] = None
            remove.append(i)
    remove = remove[::-1]
    for i in remove:
        del tdat[i]

    # average residence times on sites per site
    sdat = {d[2]:[] for d in tdat}
    for d in tdat:
        sdat[d[2]].append(d[1]-d[0])
    sdat = {k:np.mean(sdat[k]) for k in sdat}
    return(sdat)
    
def _viz_sites_and_gauss_partition():
    """ function to:
        (1) demonstrate site tags
        (2) give information about Gaussian function
    """
    exmpl = ['vac_refs/'+f for f in ['Cu111_0H2O_01', 'Cu211_0H2O_01']]
    for f in exmpl:
        surface = tuple([int(e) for e in f.split('/')[-1].split('_')[0][2:5]])
        atoms = read(f+'/POSCAR')  #_read_restart_atoms(f)
        ssites = get_sample_sites(atoms,surface=surface,repeat=(3,3))
        spos = ssites['coord']

        # report on gauss-func for weighting
        prm = list(__get_gauss_weights(ssites).values())[0]
        dmin = np.linalg.norm(spos[1:,:] - spos[0,:], axis=1).min()
        print(surface, "e^(-d^2/(2*c^2))", '| with c=%.3e'%prm, \
                                                'dmin-sites=%.3f'%dmin)
        viz_atoms = []
        # to visualize
        tags = np.unique(ssites['tags'])
        for tag in tags:
            ind = np.where(ssites['tags'] == tag)[0]
            sa = Atoms(numbers=np.ones(spos[ind,0].size)*2.0,positions=spos[ind,:],cell=atoms.get_cell())
            viz_atoms.append(atoms+sa)
        print(tags)
        view(viz_atoms)


if __name__ == "__main__":

   #_viz_sites_and_gauss_partition()
   #quit()

    folders = ["Cu211_15H2O_0%i"%i for i in range(1,5)] + ["Cu211_15H2O_OCCHO_0%i"%i for i in [1,3,4,5]]
    folders += ["Cu211_15H2O_CHO_0%i"%i for i in range(1,5)]
    folders += ["Cu211_15H2O_CO_0%i"%i for i in range(1,5)]
    folders += ["Cu211_15H2O_OH_0%i"%i for i in range(1,5)]
    folders += ["Cu211_15H2O_COH_0%i"%i for i in [1,2,3,5]]
    folders += ["Cu211_15H2O_Na+_0%i"%i for i in range(1,5)] + ["Cu211_15H2O_Na+_OCCHO_0%i"%i for i in range(1,5)]
    folders += ["Cu211_15H2O_%s_0%i"%(s,i) for s in ['Li+','K+'] for i in range(1,5)]
    folders += ["Cu211-3x4_21H2O_Na+_0%i"%i for i in range(1,5)]
    folders += ["Cu211-3x4_21H2O_Na+_OCCHO_0%i"%i for i in range(1,5)]
    folders += ["Cu211-6x4_42H2O_Na+_0%i"%i for i in range(1,5)]
    folders += ["Cu211-6x4_42H2O_Na+_OCCHO_0%i"%i for i in range(1,5)]
    folders += ["Cu111_24H2O%s_0%i"%(s,i) for s in ['','_CO','_OH'] for i in range(1,5)]
    folders += ["Au111_24H2O%s_0%i"%(s,i) for s in ['','_CO','_OH'] for i in range(1,5)]
    folders += ["Pt111_24H2O%s_0%i"%(s,i) for s in ['','_CO','_OH'] for i in range(1,5)]
    folders += ["Pt111_24H2O_05", "Pt111_24H2O_CO_05", "Cu111_24H2O_05", ]
    folders += ['%s111_24H2O%s_05'%(e,a) for e in ['Cu','Pt'] for a in ['','_CO']]
    folders += ["Cu211_15H2O-0.25q_0%i"%(i) for i in range(1,5)]
    folders += ["Cu211_15H2O-0.25q_OCCHO_0%i"%(i) for i in [1,3,4,5]]
    folders += ["Au111_24H2O_OOH_0%i"%(i) for i in [1,2,3,4]]
    folders += ["Pt111_24H2O_OOH_0%i"%(i) for i in [1,2,3,4]]
    
   #folders = ["Au111_48H2O%s_0%i"%(a,i) for a in ['','_OH'] for i in [1,2,3]]
   #folders += ["Au111_48H2O_0%i"%(i) for i in [4]]
   #folders = ['Pt111_48H2O_03', 'Pt111_48H2O_OH_02']

  ### test for weighting:
  ##f = "Cu211_15H2O_CO_01"
  # f = "Cu211_15H2O_OCCHO_03"
  ##f = "Cu111_24H2O_CO_02"
  ##f = "Cu111_24H2O_OH_01"
  # site_dat = _load_pickle_file(f+'/site_sampling.pkl')

  # stag = f.split('/')[-1].split('_')[0]
  # atag = __get_adstag_fname(f.split('/')[-1])
  # #straj = _get_ads_site_traj(site_dat, stag, atag, reduced=False, gaussian_weighted=False)
  # straj = _get_ads_site_traj(site_dat, stag, atag, reduced=True, gaussian_weighted=True)
  # print(_traj_overview(f,site_dat))
  # 
  # # test as in post_reoptimized
  # winds = {ss:(np.isnan(straj[ss]) == False).nonzero()[0] for ss in straj}
  # winds = {ss:winds[ss][np.where(straj[ss][winds[ss]] > 0.50)[0]] for ss in straj}
  # for site in winds: # to check site weights
  #     print(site,winds[site].shape)
  #######################
  
    ### test for CO desorbtion/ CHO dissociation on Cu211
    #f = "Au111_24H2O_CO_02"
  # f = "Cu211_15H2O_CHO"
  # # FOR NOW
  # for f in folders:
  #     print(f)
  #     site_traj_data = _post_surface_sampling(f)
  #     _write_pickle_file(f+'/site_sampling.pkl', site_traj_data)
  # assert False
    ########
  # site_dat = _load_pickle_file(f+'/site_sampling.pkl')
  # stag = f.split('/')[-1].split('_')[0]
  # atag = __get_adstag_fname(f.split('/')[-1])
  # #straj = _get_ads_site_traj(site_dat, atag, reduced=False, gaussian_weighted=False)
  # straj = _get_ads_site_traj(site_dat, stag, atag, reduced=True, gaussian_weighted=True)
  # print(_traj_overview(f,site_dat))
    #####################################################
    #assert False
    ## end of test
    
    for f in folders:
        fpkl = '%s/site_sampling.pkl'%f
        if not os.path.isfile(fpkl):
            print('sampling %s'%f)
            site_traj_data = _post_surface_sampling(f)
            _write_pickle_file(fpkl, site_traj_data)

    ###########################################################################
    # special outsorting trajs: Pt111_24H2O_OOH_03/04 because of Pt-layer shift
    manout = {'Pt111_24H2O_OOH_03':np.array(range(20000,33061)), 'Pt111_24H2O_OOH_04':np.array(range(10000,33101))}
    for f in manout:
        print('ammend site-traj %s'%f)
        fpkl = '%s/site_sampling.pkl'%f
        site_dat = _load_pickle_file(fpkl)
        site_dat[0].update({'manual-outsort':manout[f]})
        _write_pickle_file(fpkl, site_dat)
    ###########################################################################

    
    # write info file
    trajtxt = ''; trajdtl = ''
    for f in folders:
        print(f)
        site_dat = _load_pickle_file(f+'/site_sampling.pkl')
        trajtxt += _traj_overview(f,site_dat)
        trajdtl += _detailed_traj_output(f,site_dat)
    with open('output/traj_overview.txt','w') as f_out:
        f_out.write(trajtxt)
    with open('output/traj_details.txt','w') as f_out:
        f_out.write(trajdtl)
    raise Exception('did that work?')
    if True:
        print(f)
        straj = get_ads_site_traj(f)
        #analyze_surface_interaction(awdict) #NOT sure what this is for

    # script:
    #         TODO: make plotting scripts #1! charging? - also scatter plot
    #         TODO: add Bader analysis to check for total charge in slab


    #### NOTE: discretization works fine - only "fails" (does what it should)
    ####       when averting from initial (only nH2O) solvent configuration
    ####       checks for Cu211-OH/COH show correct behavior everything non-right (strong-solvent interaction) is thrown out
    
