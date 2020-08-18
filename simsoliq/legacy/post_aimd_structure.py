#!/usr/bin/env python

import sys, os
import numpy as np
import hashlib, base64
from time import time

sys.path.append('/home/cat/heenen/Workspace/TOOLS/ASE-DTU/ase')
from ase.data import chemical_symbols as symbls
from ase.visualize import view
from ase.io import read

from operator import itemgetter
from itertools import *

sys.path.append("/home/cat/heenen/Workspace/benchmarking_calcs/Cu_AIMD_runs/scripts_vasp")
from post_process_traj import _read_restart_atoms, _sort_sub_runs, plot_msd_individual, plot_diffusion_coefficients, _plot_orientation_distribution, _get_ledge_metal_slab
from transform_lattice_constant_standalone import identify_water
from md_tools import atomic_diffusion_tools as mdtool
sys.path.append("/home/cat/heenen/Workspace/benchmarking_calcs/Cu_AIMD_runs/reoptimize_RPBE-D3")
from post_reoptimized import _get_aimd_folders

from test_site_occupation import get_sample_sites
        
from test_site_occupation import get_ads_site_traj
from cathelpers.misc import _load_pickle_file, _write_pickle_file
from cathelpers.atom_manipulators import _correct_vec, get_type_atoms, get_CN, get_type_indices



backup_ainds = {
    "Pt111_24H2O_CO_01":[48,73],
    "Cu211-6x4_42H2O_Na+_OCCHO_03":[0,81,82,83,84],\
    "Cu211-6x4_42H2O_Na+_OCCHO_04":[0,83,84,85,86],\
    }
backup_ainds.update({"Cu211_15H2O_Na+_OCCHO_0%i"%i:[0,31,32,33,34] \
    for i in range(1,5)})
backup_ainds.update({"Cu211-3x4_21H2O_Na+_OCCHO_0%i"%i:[0,43,44,45,46]\
    for i in range(1,5)})
backup_ainds.update({"Cu211-6x4_42H2O_Na+_OCCHO_0%i"%i:[0,85,86,87,88]\
    for i in range(1,3)})
        
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
        
def _separate_top_layer(atoms, oinds):
    inds = {i:0 for i in oinds}
    for i in range(len(atoms)):
        Oz = atoms[i].get_positions()[oinds,2]
        ind = oinds[np.where(Oz > Oz.max()-4.0)[0]]
        for ii in ind:
            inds[ii] += 1
    inds = {ii:inds[ii] / len(atoms) for ii in inds}
    #ind_out = [i for i in inds if inds[i] < 0.15]
    #ind_out = [i for i in inds if inds[i] < 0.01]
    ind_out = [i for i in inds if inds[i] < 0.05]
    return(np.array(ind_out, dtype=int))
    
def _get_msd_folder(f):
    fpkl = f+"/msd_dict.pkl"
    if not os.path.isfile(fpkl):
        print(f)
        ainds, winds = _get_water_adsorbate_inds(f)

        atoms = _read_restart_atoms(f)
        atoms = atoms[3000:] # cut first 3 ps
        oinds = winds[np.where(atoms[0].get_atomic_numbers()[winds] == 8)[0]]

        oinds = _separate_top_layer(atoms[::100], oinds)
        
        omsd = mdtool._calc_msd_time_averaged(atoms[::100],inds={'x':oinds},lag=1.0)
        omsd_av = mdtool._calc_msd_time_averaged(atoms[::100],inds={'x':oinds},lag=0.5)

        amsd = {'x':np.zeros(omsd['x'].shape)}; amsd_av = {'x':np.zeros(omsd_av['x'].shape)}
        if len(ainds) > 0:
            amsd = mdtool._calc_msd_time_averaged(atoms[::100],inds={'x':ainds},lag=1.0)
            amsd_av = mdtool._calc_msd_time_averaged(atoms[::100],inds={'x':ainds},lag=0.5)
        _write_pickle_file(fpkl,{'water':{'av0.5':omsd_av['x'], 'msd':omsd['x']},\
                                   'ads':{'av0.5':amsd_av['x'], 'msd':amsd['x']}})
    d_msd = _load_pickle_file(fpkl)
    return(d_msd)

def evaluate_diffusion_coefficients(folders, d_msd, msd_excl={}):
    # pre-sort
    fs = _sort_sub_runs(folders)
    # make plots + evaluate diffusion coefficients xy
    fs = {f:fs[f] for f in fs if np.all([f.find(m) == -1 for m in ['Li+','Na+','K+']])}
    dfcs = {}
    msdk = ['x','y','z','xy','xz','yz','xyz']; d={'water':3, 'ads':2}
    for f in fs:
        dfc = {'ads':[], 'water':[]}
        for ff in fs[f]:
            if ff not in msd_excl:
                for dd in ['ads','water']:
                    dat = d_msd[ff][dd]['av0.5'][:,msdk.index('xy')]
                    t = np.arange(dat.size)*0.1 #ps
                    dfc[dd].append(((dat[-1]-dat[0]) / (t[-1]-t[0])) * (1./(2*d[dd])))
        dfcs.update({f:dfc})
   #A2ps = 1e-20*1e9
   #for f in dfcs:
   #    print(f, np.mean(dfcs[f]['water'])*A2ps)
   #for f in dfcs:
   #    print(f, np.mean(dfcs[f]['ads'])*A2ps)
    plot_diffusion_coefficients('msd/diffcoef_overview', dfcs, write_eps=True)

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

def _make_hash(array):
    array[np.where(array == 0)] = 0
    hashed = hashlib.md5(array).digest()
    hashed = base64.urlsafe_b64encode(hashed)
    return(hashed)

def _eval_hbonds_traj(a, dat):
    # evaluate H-bonds: O-O distances / cutoff 3.5 AA / O-H-O angle > 140*
    dHb = []
    cell = a[0].get_cell()
    for i in range(len(a)):
        dd = [dat[hsh] for hsh in dat if i in dat[hsh]['tinds']]
        if len(dd) > 0: # only if valid solvent structure
            dd = dd[0]
            nOHb = 0 # count for H-bonds
            atno = a[i].get_atomic_numbers()
            spos = a[i].get_scaled_positions()
            # Os in adsorbate and water
            octs = [ii for ii in dd['ainds'] if atno[ii] == 8]
            ocdn = [ii for ii in dd['winds'] if atno[ii] == 8]
            inOHb = [0]*len(octs) # count for individual H-bonds per O
            Hbcnd = {}
            for oo in range(len(octs)): #oc in octs: #iterate over o-center / find possibel coordinating
                oc = octs[oo]
                nOHib = 0 # bonds per ads-O
                # check for distances to o-water
                rvec = _correct_vec(spos[ocdn,:] - spos[oc,:])
                dvec = np.dot(rvec,cell)
                d = np.linalg.norm(dvec,axis=1)
                indc = np.array(ocdn)[np.where(d < 3.5)[0]]
                # check for ads-OH
                ah = [ii for ii in dd['ainds'] if atno[ii] == 1 and \
                    np.linalg.norm(np.dot(_correct_vec(spos[ii,:]-spos[oc,:]),cell)) < 1.4]
                if len(ah) > 1:
                    print(dd['ainds'])
                    view(a[i-100:i+100])
                    raise Exception('*OH2 found?')
                # in principle angle > 140* should be enough (needs to be inbetween O-O)
                for io in indc:  # iterate water-o
                    for ih in dd['wts'][io]:  # iteratre water-h
                        a_oho = a[i].get_angle(oc,ih,io,mic=True)
                        dO = np.linalg.norm(np.dot(_correct_vec(spos[io,:] - spos[oc,:]),cell))
                        boh = np.linalg.norm(np.dot(_correct_vec(spos[ih,:] - spos[oc,:]),cell))
                        if a_oho > 140: # and boh < 2.0:
                            if io not in Hbcnd or io in Hbcnd and Hbcnd[io][0] > dO: # overwrite if closer
                                nOHib += 1
                                # save data of saved angle
                                Hbcnd.update({io:[oc, dO, a_oho]})
                        #print(oc, io, ih, a[i].get_angle(oc,ih,io,mic=True))
                    # iterate ads-h
                    for ih in ah:
                        a_oho = a[i].get_angle(oc,ih,io,mic=True)
                        dO = np.linalg.norm(np.dot(_correct_vec(spos[io,:] - spos[oc,:]),cell))
                        boh = np.linalg.norm(np.dot(_correct_vec(spos[ih,:] - spos[io,:]),cell))
                        if a_oho > 140: # and boh < 2.0:
                            # check if o previously saved:
                            if io not in Hbcnd or io in Hbcnd and Hbcnd[io][0] > dO: # overwrite if closer
                                nOHib += 1
                                # save data of saved angle
                                Hbcnd.update({io:[oc, dO, a_oho]})
                        #print(oc, io, ih, a[i].get_angle(oc,ih,io,mic=True))
               #if nOHib > 2:
               #    view(a[i])
               #    raise Exception('too many H-bonds? see structure')
         #      nOHb += nOHib
         #      inOHb[oo] += nOHib
         #  dHb.append([i,nOHb]+inOHb)
            # only where the conditions are met
           #print(Hbcnd)
           #print(octs)
           #print([Hbcnds for Hbcnds in Hbcnd])
            subHb = {ao:[Hbcnds for Hbcnds in Hbcnd if Hbcnd[Hbcnds][0] == ao] for ao in octs}
           #print(subHb)
            subHb = [len(subHb[ao]) for ao in np.sort(octs)]
           #print(subHb)
            dHb.append([i,sum(subHb)]+subHb)
           #assert False
    return(np.array(dHb))

def evaluate_hbond_count(folder):
    t0 = time()
    # read atoms - determine inds of system
    a = _read_restart_atoms(folder)
    ainds0, winds0 = _get_water_adsorbate_inds(folder) # water-str. guess
    winds0 = [i for i in winds0 if a[0].get_chemical_symbols()[i] in ['H','O']] # remove Na+ from water
    dat = eval_sys_inds(a, ainds0, winds0) # water-str evaluation
    # evaluate h-bonds (O-H-O) O/H ads - O/H water
    dhbond = _eval_hbonds_traj(a, dat)
    print('evaluating h-bonds %s took %.1f s'%(f,(time()-t0)))
    return(dhbond)

def _get_h2o_orientation(a, dat, slim=5000):
    # least data-intensive way -- begin sampling at 5000
    # save data within bins 
    hbins = np.linspace(0,a[0].get_cell()[2,2],200) # hist by height
    centr = (hbins[1:]+hbins[:-1])/2.0
    wh2o_dat = {ct:[] for ct in centr} # data for h2o molecular orienation
    woh_dat = {ct:[] for ct in centr} # data for O-H bond orientation

    cell = a[0].get_cell()
    # evaluate 2 types of angles (molecular H2O & O-H bonds)
    for i in range(slim,len(a)):
        dd = [dat[hsh] for hsh in dat if i in dat[hsh]['tinds']]
        if len(dd) > 0: # only if valid solvent structure
            dd = dd[0]
            atno = a[i].get_atomic_numbers()
            spos = a[i].get_scaled_positions()
            # iterate waters
            for io in dd['wts']:
                # z-value
                z = np.dot(spos[io,:], cell)[2]
                zb = centr[np.where((hbins - z) > 0)[0][0]-1] # z lies under this bin
                
                # (1) molecular axis
                im = np.array([spos[ih,:] for ih in dd['wts'][io]])
                dim = _correct_vec(im[0,:] - im[1,:])/2.0
                im = im[1,:]+dim
                vh2o = _correct_vec(im-spos[io,:])
                agl = np.arccos(np.dot(vh2o,np.array([0,0,1]))/(np.linalg.norm(vh2o)))*(180./np.pi)
                wh2o_dat[zb].append(agl)
                
                # (2) O-H bonds (easier) - harder statistics? -> distribution binned --> plot z bin vs angle!!!
                for ih in dd['wts'][io]:
                    v_oh = _correct_vec(spos[ih,:]-spos[io,:])
                    agl = np.arccos(np.dot(v_oh,np.array([0,0,1]))/(np.linalg.norm(v_oh)))*(180./np.pi)
                    woh_dat[zb].append(agl)
    return(wh2o_dat, woh_dat)
        
def _evaluate_water_orientation(f):
    t0 = time()
    # read atoms object 
    a = _read_restart_atoms(f)
    # determine system inds
    ainds0, winds0 = _get_water_adsorbate_inds(f)
    dat = eval_sys_inds(a, ainds0, winds0)
    o_h2o, o_oh = _get_h2o_orientation(a, dat, slim=slim)
    print('evaluating h2o-orientation %s took %.1f s'%(f,(time()-t0)))
    return(o_h2o, o_oh)
    
#def _evaluate_h2o_adsorption(f, dchs=2.3):
#    a = _read_restart_atoms(f)
#    dsm = 1.0; #dchs = 2.3 # chemisorption distance (metal-atom-O distance)
#    cell = a[0].get_cell()
#   
#    # get sites for surface
#    rdict = {'111':[3,4], '211':[3,3]}
#    site_dict = get_sample_sites(a[0],surface=[int(i) for i in \
#                f.split('_')[0][2:5]],repeat=rdict[f.split('_')[0][2:5]])
#    ssites = site_dict['scaled_coord']
#    tagme, me_top = _sort_top_me_atoms(f, a[0], site_dict)
#    
#    # pre-determine indeces for water 
#    ainds0, winds0 = _get_water_adsorbate_inds(f)
#    winds0 = [i for i in winds0 if a[0].get_chemical_symbols()[i] in ['H','O']] # remove Na+ from water
#    dat = eval_sys_inds(a, ainds0, winds0)
#
#    # data-format twisted: site-based
#    d = {s:[] for s in range(len(site_dict['tags']))}
#    
#    count = 0
#
#    for i in range(0,len(a)):
#        # choose right adsorbate configuration
#        dd = [dat[hsh] for hsh in dat if i in dat[hsh]['tinds']]
#        if len(dd) > 0: # only if valid solvent structure
#            dd = dd[0]
#            spos = a[i].get_scaled_positions()
#            sme = spos[me_top,:]
#            for io in dd['wts']:
#                # distance to sites
#                dist, mind = __get_distances_order(ssites, spos[io,:], cell) 
#                dind = mind[np.where(dist < dchs-dsm)[0]]
#                if len(dind) > 0:
#                    min_ind = dind[0]
#                    d[min_ind].append([i,io])
#                #### - only append smallest distance
#                #[d[s].append([i,io]) for s in dind] # s=site; i=time/ind; io=water ind
#                ####
#                # distance to metal centers
#                dist, mind = __get_distances_order(sme, spos[io,:], cell) 
#                dind = mind[np.where(dist < dchs)[0]]
#                if len(dind) > 0 and (len(d[tagme[dind[0]]]) == 0 or \
#                    not np.array_equal(d[tagme[dind[0]]][-1], [i,io])):
#                    min_ind = dind[0]
#                    d[tagme[min_ind]].append([i,io])
#            #view(a[i] + Atoms('He'*dist.size, site_dict['coord']))
#    for ste in d:
#        d[ste] = np.array(d[ste])
#    dout = {'tags':site_dict['tags'], 'tlen':len(a), 'sh2o':d}
#    return(dout)
        
#def _sort_top_me_atoms(f, a, site_dict):
#    ssites = site_dict['scaled_coord']
#    spos = a.get_scaled_positions()
#    atno = a.get_atomic_numbers()
#    try:
#        ifix = a.constraints[0].index
#    except IndexError: #somehow not fix on atoms object
#        b = read(f+'/POSCAR')
#        ifix = b.constraints[0].index
#    ime = np.unique(atno[ifix])[0]
#
#    me_top = [j for j in range(len(a)) if atno[j] == ime and j not in ifix]
#    sme = spos[me_top,:]
#    tagme = []
#    for j in range(sme[:,0].size):
#        dmests = np.linalg.norm(_correct_vec(ssites - sme[j,:]),axis=1).argsort()
#        for jj in dmests:
#            if site_dict['tags'][jj].split('-')[0] == 'top':
#                tagme.append(jj)
#                break
#    return(tagme, me_top)

def __get_distances_order(parray, pt, cell): 
    dvec = np.dot(_correct_vec(parray - pt),cell)
    dist = np.linalg.norm(dvec, axis=1)
    mind = dist.argsort(); dist = dist[mind] # only append minimum
    return(dist, mind)
    
def _reduce_sequence_array(dat):
    ndat = []
    for i in range(1,np.unique(dat[:,1]).size+1): #iterate states
        cons = np.where(dat[:,1] == i)[0]
        lpart = [list(map(itemgetter(1), g)) for k, g in \
                groupby(enumerate(cons), lambda x: x[0]-x[1])]
        ndat += [dat[lp[-1],:] for lp in lpart]
    ndat = np.array(ndat)
    ndat = ndat[np.argsort(ndat[:,0]),:]
    return(ndat)
    
def _estimate_adsorbate_exchange(dat): 
    cads = 1; cdat = []
    # isolate residence times of adsorbate composition
    for shsh in dat:
        tinds = dat[shsh]['tinds']; ainds = dat[shsh]['ainds']
        lpart = [list(map(itemgetter(1), g)) for k, g in \
                groupby(enumerate(tinds), lambda x: x[0]-x[1])]
        for lp in lpart:
            cdat.append([lp[0], cads])
        cads += 1
    cdat = np.array(cdat)
    cdat = cdat[np.argsort(cdat[:,0]),:]
    
    cdatc = _reduce_sequence_array(cdat)
    # eliminate too short stays of adsorbate lim = 150 ps
    b = 0; nch = 0; rcdatc = [cdatc[0,:]]
    for i in range(1,cdatc[:,0].size):
        #print(cdatc[i,:], cdatc[i,0] - cdatc[i-1,0])
        if cdatc[i,0] - cdatc[i-1,0] > 150:
            rcdatc.append(cdatc[i,:])
    rcdatc = np.array(rcdatc)
    cdatc = _reduce_sequence_array(rcdatc)

    return(max(np.unique(cdat[:,1]).size-1, cdatc[:,0].size-1))
                

def _split_h2o_adsorption(sdat):
    events = []
    # split (1) water atoms and (2) time sequence
    for w in np.unique(sdat[:,1]): #(1)
        ind = np.where(sdat[:,1] == w)[0] #only w entries
        lpart = [list(map(itemgetter(1), g)) for k, g in \
            groupby(enumerate(sdat[ind,0]), lambda x: x[0]-x[1])]
        # time sequence bounds
        lpart = [[lp[0],lp[-1]] for lp in lpart]
        lp0 = [lpart[0]]
        for l in range(1, len(lpart)):
            if lpart[l][0] - lp0[-1][1] < 150:
                lp0[-1][1] = lpart[l][1]
            else:
                lp0.append(lpart[l])
        events += lp0
    return(events)

        
def _print_table(nsites, tsites, tlen):
    strout = ""
    sts = list(nsites.keys()) #tags
    # set up data-array
    d = np.array([[nsites[s], len(tsites[s]), np.mean(tsites[s])] for s in nsites])
    # kick out zero-value entries
    indn0 = np.where(d[:,0] != 0.0)[0]; sts = [sts[i] for i in indn0]; d = d[indn0,:]
    # reorder 
    ind = np.argsort(d[:,0])[::-1]; sts = [sts[i] for i in ind]; d = d[ind,:]
    # add total
    sts = ['total']+sts
    tot = [d[:,0].sum(), d[:,1].sum(), __average_freqs(d)]
    d = np.array([tot]+[d[i,:].tolist() for i in range(d[:,0].size)])
    d[:,1] /= (tlen*1e-3) # make frequency / ps
    for i in range(d[:,0].size):
        dstr = [sts[i], "%.2f"%np.around(d[i,0],2), "%.4f"%d[i,1], "%.1f"%np.around(d[i,2],1)]
        strout += " & ".join(dstr) + "  \\\\ \n"
    return(strout)

def __average_freqs(d):
    fqs = []
    for i in range(d[:,0].size):
        fqs += int(d[i,1])*[d[i,2]]
    return(np.mean(fqs))


if __name__ == "__main__":

    mdtool = mdtool()
    folders = _get_aimd_folders()
    #folders = ['Au111_48H2O%s_0%i'%(a,i) for a in ['','_OH'] for i in [1,2]]
   #folders = ['Au111_48H2O_0%i'%(i) for i in [1,2,3,4]]
   #folders += ['Au111_48H2O_OH_0%i'%(i) for i in [1,2,3]]
   #folders += ['Au111_24H2O_0%i'%i for i in range(1,5)]
   #folders += ['Au111_24H2O_OH_0%i'%i for i in range(1,5)]
   #folders = ['Pt111_48H2O_03', 'Pt111_48H2O_OH_02']
   #folders += ['Pt111_24H2O_OH_0%i'%i for i in range(1,5)]
    folders = ["Cu211_15H2O_0%i"%(i) for i in range(1,5)]
    folders += ["Cu211_15H2O%s_0%i"%(a,i) for a in ['_OCCHO'] for i in [3,4,5]]
    folders += ['Cu211_15H2O_%s_0%i'%(a,i) for a in ['Na+_OCCHO','Na+'] for i in range(1,5)]
    #folders += ['Cu211_15H2O_%s_0%i'%(a,i) for a in ['Na+_OCCHO','Na+','Li+','K+'] for i in range(1,5)]
    folders += ["Cu211-3x4_21H2O%s_0%i"%(a,i) for a in ['_Na+_OCCHO','_Na+',''] for i in range(1,5)]
    folders += ["Cu211-3x4_21H2O_OCCHO_0%i"%(i) for i in [1,2,4]]
    folders += ["Cu211-6x4_42H2O_%s_0%i"%(a,i) for a in ['Na+_OCCHO','Na+'] for i in range(1,5)]
    folders += ["Cu211_15H2O-0.25q_0%i"%(i) for i in range(1,5)]
    folders += ["Cu211_15H2O-0.25q_OCCHO_0%i"%(i) for i in [1,3,4,5]]
    folders += ["Cu211_15H2O-0.50q%s_0%i"%(a,i) for a in ['', '_OCCHO'] for i in range(1,5)]
    folders.remove("Cu211_15H2O-0.50q_OCCHO_02") # messed up pseudopotentials
    
    # test what is up with ions
  # folders = ['Cu211_15H2O_Na+_OCCHO_02']
    #folders = ['Cu211_15H2O_Na+_02']
    #folders = ['Cu211_15H2O_Na+_0%i'%i for i in range(1,5)]
   #folders = ['Cu211_15H2O_Na+_OCCHO_0%i'%i for i in range(1,5)]
    #folders = ["Cu211-3x4_21H2O_Na+_OCCHO_0%i"%(i) for i in [1,3,2,4]]
    #folders = ["Cu211-6x4_42H2O_Na+_OCCHO_0%i"%(i) for i in [1,2,3,4]]

    # TODO: seriously make unified factory function with pkl-file
    #       data (use decorators)

    ########### get *OH<->H2O converstion frequency ############
    foh = [f for f in folders if 'OH' in f.split('_')]
    for f in foh:
        a = _read_restart_atoms(f)
        # determine system inds
        ainds0, winds0 = _get_water_adsorbate_inds(f)
        dat = eval_sys_inds(a, ainds0, winds0)
        nchange = _estimate_adsorbate_exchange(dat)
        print(f, len(a), 'changes: %i'%nchange)
    
    assert False
    ############################################################
    
    ################ get H2O adsorption events #################
    #ddchs = {'Pt111':2.48, 'Au111':2.69, 'Cu111':2.3, 'Cu211':2.3} # RESULTS appear unreasonable
    ddchs = {'Pt111':2.55, 'Au111':2.55, 'Cu111':2.55, 'Cu211':2.55, 'Cu211-3x4':2.55, 'Cu211-6x4':2.55}; tstart = 5000.
    wadat = {}
    for f in folders:
        pklfile = f+'/h2o_adsorption.pkl'
        if not os.path.isfile(pklfile):
            print('evaluating %s'%f); t0 = time()
            dads = _evaluate_h2o_adsorption(f, dchs=ddchs[f.split('_')[0]])
            print('took %.1f s'%(time() - t0))
            _write_pickle_file(pklfile, dads)
        dat = _load_pickle_file(pklfile)
        # correct for dissociation or desorption
        wtraj = get_ads_site_traj(f, reduced=True, gaussian_weighted=True)
        if len(wtraj) > 0:
            tind = np.where(np.isnan(wtraj[list(wtraj.keys())[0]]) == False)[0]
            for ste in dat['sh2o']:
                d = np.array(dat['sh2o'][ste]) # can go out if pkl file written new
                if d.size > 0:
                    ind = [i for i in range(d[:,0].size) if d[i,0] in tind]
                    d = d[ind,:]
                dat['sh2o'][ste] = d
        wadat.update({f:dat})

    fs = _sort_sub_runs(folders)
    strout = []; plotout = {}
    for fbase in fs:
        # site dict
        tlen = 0
        nsites = {stype:0 for stype in wadat[fs[fbase][0]]['tags']}
        tsites = {stype:[] for stype in wadat[fs[fbase][0]]['tags']}
        nadsruns = []
        for f in fs[fbase]:
            tlen += wadat[f]['tlen'] - tstart
            for s in wadat[f]['sh2o']:
                tag = wadat[f]['tags'][s]
                sdat = np.array(wadat[f]['sh2o'][s])
                if len(wadat[f]['sh2o'][s]) > 0:
                    indstrt = np.where(sdat[:,0] > tstart)[0]
                    sdat = sdat[indstrt,:]
                # count for traj share of adsorbtion
                nsites[tag] += len(sdat)
                # determination of frequency
                if sdat.size > 0:
                    avnts = _split_h2o_adsorption(sdat)
                    tsites[tag] += [av[1]-av[0] for av in avnts]
           #rough_sum = {s:np.sum(tsites[s]) for s in tsites} # not actually adsorbed - 150s added
           #print(f, sum([rough_sum[k] for k in rough_sum]) / wadat[f]['tlen'], rough_sum)
            nadsruns.append(sum([len(np.array(wadat[f]['sh2o'][s])[np.where(np.array(wadat[f]['sh2o'][s])[:,0] > tstart)[0],0]) for s in wadat[f]['sh2o'] if len(wadat[f]['sh2o'][s]) > 0]) / (wadat[f]['tlen']-tstart))
        print(fbase, nadsruns, np.mean(nadsruns), np.std(nadsruns)) #for std
        plotout.update({fbase:[np.mean(nadsruns), np.std(nadsruns)]})
        nsites = {s:nsites[s]/tlen for s in nsites}
        strout.append(fbase+"\n"+_print_table(nsites, tsites, tlen))
    with open('output/h2o_adsorption.txt','w') as fout:
    with open('output/h2o_adsorption_ions.txt','w') as fout:
        fout.write('\n\n'.join(strout))
    pklfile = 'output/h2o_adsorption_plot.pkl'
    _write_pickle_file(pklfile, plotout)

    ############################################################

    ##################### H2O orientation ######################
    slim = 5000
    odat = {}
    for f in folders:
        pklfile = f+'/h2o_orientation.pkl'
        if not os.path.isfile(pklfile):
            print('evaluating %s'%f)
            o_h2o, o_oh = _evaluate_water_orientation(f)
            _write_pickle_file(pklfile, {'h2o':o_h2o, 'OH':o_oh})
        odat.update({f:_load_pickle_file(pklfile)})
    
    fs = _sort_sub_runs(folders)
    for fbase in fs:
        # unify statistics
        zmetal = np.mean([_get_ledge_metal_slab(f, f[:2]) for f in fs[fbase]])
        dat = {sub:{bn:[] for bn in odat[fs[fbase][0]][sub]} for sub in odat[fs[fbase][0]]}
        for sub in odat[fs[fbase][0]]:
            for f in fs[fbase]:
                for bn in odat[f][sub]:
                    dat[sub][bn] += odat[f][sub][bn] 
        # plot - orientation plot per fbase
        _plot_orientation_distribution('water_orientation_%s'%fbase, dat, z0=zmetal)
    ############################################################

    ################### H-bond evaluation #####################
   ##folders = ['Au111_24H2O_OOH_01', 'Cu111_24H2O_OH_01', 'Pt111_24H2O_OOH_01']
    tstart = 5000
    dhdat = {}
    for f in folders:
        dfile = f+'/chbond_traj.txt'
        if not os.path.isfile(dfile):
            print('evaluating %s'%f)
            dhbond = evaluate_hbond_count(f)
            np.savetxt(dfile, dhbond)
        if os.path.getsize(dfile) > 0:
            dhbond = np.loadtxt(dfile)
            # eval discretization to capture dissociation/desorption inds 
            wtraj = get_ads_site_traj(f, reduced=True, gaussian_weighted=True)
            if len(wtraj) > 0:
                tind = np.where(np.isnan(wtraj[list(wtraj.keys())[0]]) == False)[0]
                ind = [i for i in range(dhbond[:,0].size) if dhbond[i,0] in tind]
                dhbond = dhbond[ind,:]
            dhdat.update({f:dhbond})

    # for making statistic output (table)
    fs = _sort_sub_runs(folders)
    for fbase in np.sort(list(fs.keys())):
        dhbonds = []; dhbonderr = []
        for fr in fs[fbase]:
            if fr in dhdat:
                dhbond = dhdat[fr]
                indstrt = np.where(dhbond[:,0] > tstart)[0]
                dhbonds.append(np.mean(dhbond[indstrt,1]))
                dhbonderr.append(np.std(dhbond[indstrt,1]))
                dhbondsep = [np.mean(dhbond[indstrt,i]) for i in range(1,dhbond[0,:].size)]
                dhbondseperr = [np.std(dhbond[indstrt,i]) for i in range(1,dhbond[0,:].size)]
                print(fr, dhbondsep, dhbondseperr)
        print(fbase, dhbonds, np.mean(dhbonds), 'p/m', np.std(dhbonds), '/', np.mean(dhbonderr))
    ###########################################################


    #################### evaluate MSD data ####################
    d_msd = {}
    for f in folders:
        d_msd.update({f:_get_msd_folder(f)})

    # pre-sort
    fs = _sort_sub_runs(folders)
    # make msd plots
    #for f in fs:
    for f in ['Cu211_15H2O_CHO']:
        plot_msd_individual('msd/msd_%s_overview'%f, {ff:d_msd[ff] for ff in fs[f]}, write_eps=True)
    # OH --> somehow the conversion to H2O is not correctly accounted for, Au111/CO --> desorption
    msd_excl = {'Cu111_15H2O_OH_01':None, 'Au111_24H2O_CO_03':[23234,30157], 'Au111_24H2O_CO_04':None,\
        'Pt111_24H2O_OOH_02':None, 'Pt111_24H2O_OOH_03':None, 'Pt111_24H2O_OOH_04':None} #hardcoded
    evaluate_diffusion_coefficients(folders, d_msd, msd_excl)
    ###########################################################

