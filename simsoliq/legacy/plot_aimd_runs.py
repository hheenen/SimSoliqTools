#!/usr/bin/env python

import sys, os
import numpy as np
sys.path.append("/home/cat/heenen/Workspace/benchmarking_calcs/Cu_AIMD_runs/scripts_vasp")
from post_process_traj import get_OUTCAR_energetics, plot_engs_vs_time, plot_av_engs_vs_time, _get_density, plot_density, _file_age_check, _get_time_latest_OUTCAR, plot_av_dE_vs_time, get_vasp_energetics, _get_time_latest_vaspout, get_run_aimd_wf, plot_wf_vs_time, get_run_dvac_sp, plot_esol_vs_time, _process_densities, _sort_sub_runs, get_linear_fit, _running_av, _plot_running_av_overview, _plot_h2o_adsorption, _av_dens, plot_density_fancy
sys.path.append("/home/cat/heenen/Workspace/benchmarking_calcs/Cu_AIMD_runs/runs_AIMD/gas_refs")
from ref_engs import get_energies
from ase.data import chemical_symbols
from cathelpers.misc import lines_key, _load_pickle_file, _write_pickle_file


def eval_aimd_rel_energies(dats):
   #aref = {}
   #refOCCHO = -32.39397
   #refvibrotOCCHO = 0.09697
   #reftransOCCHO = 0.09694
   #kinOCCHO = refOCCHO + refvibrotOCCHO + reftransOCCHO
   #potOCCHO = refOCCHO
   #aref['OCCHO'] ={'epot':potOCCHO, 'etot':kinOCCHO}
   #
   #refCHO = -17.94150; refvibrotCHO = 0.05818; reftransCHO = 0.05817
   #aref['CHO'] ={'epot':refCHO, 'etot':refCHO+refvibrotCHO+reftransCHO}

   #refCO = -14.45248; refvibrotCO = 0.03879; reftransCO = 0.03878
   #aref['CO'] ={'epot':refCO, 'etot':refCO+refvibrotCO+reftransCO}

   #refOH = -10.66722; refvibrotOH = 0.03879; reftransOH = 0.03878 # 1/2 H2 1/2 O2 (but ref refvibrotOH to H2O?) running O2!
   #aref['OH'] ={'epot':refOH, 'etot':refOH+refvibrotOH+reftransOH}
    aref = make_refs(dat,['OCCHO', 'CHO', 'CO', 'COH', 'OH'])

    ctypes = {'Cu211_15H2O_OCCHO':["Cu211_15H2O_0%i","Cu211_15H2O_OCCHO_0%i",range(1,5),[1,3,4,5]],\
             'Cu211_15H2O_Na+_OCCHO':["Cu211_15H2O_Na+_0%i","Cu211_15H2O_Na+_OCCHO_0%i",range(1,5),range(1,5)],\
             'Cu211-3x4_21H2O_Na+_OCCHO':["Cu211-3x4_21H2O_Na+_0%i","Cu211-3x4_21H2O_Na+_OCCHO_0%i",range(1,5),range(1,5)],\
             'Cu211_15H2O_CHO':["Cu211_15H2O_0%i","Cu211_15H2O_CHO_0%i",range(1,5),range(1,5)],\
             'Cu211_15H2O_COH':["Cu211_15H2O_0%i","Cu211_15H2O_COH_0%i",range(1,5),[1,2,3,5]],\
             'Cu211_15H2O_CO':["Cu211_15H2O_0%i","Cu211_15H2O_CO_0%i",range(1,5),range(1,5)],\
             'Cu211_15H2O_OH':["Cu211_15H2O_0%i","Cu211_15H2O_OH_0%i",range(1,5),range(1,5)],\
             }

    for ctyp in ctypes:
        print(ctyp)
        ref = aref[ctyp.split('_')[-1]]
        # print binding energies:
        for etype in ['epot','etot']:
            # 
            e_ref = np.array([np.mean(dats[f][etype][-10000:]) for f in [ctypes[ctyp][0]%i for i in ctypes[ctyp][2]]])
            print('Cu211_ref',np.mean(e_ref),np.std(e_ref),'energies-all:',e_ref)
            e_ads = np.array([np.mean(dats[f][etype][-10000:]) for f in [ctypes[ctyp][1]%i for i in ctypes[ctyp][3]]])
            print('Cu211_ads',np.mean(e_ads),np.std(e_ads),'energies-all:',e_ads)
            print('binding OCCHO in 15H2O: mean %.3f, allmin %.3f, range %.3f-%.3f'\
                %(np.mean(e_ads)-np.mean(e_ref)-ref[etype],e_ads.min()-e_ref.min()-ref[etype],\
                e_ads.max()-e_ref.min()-ref[etype],e_ads.min()-e_ref.max()-ref[etype]))
        
def _post_sort_process_aimd_data(dats, inds, weights=None, icut=5000):
    # prep weights:
    if weights == None: #weight everything 1.0
        weights = {f:{ss:np.ones(len(dats[f]['epot'])) for ss in inds[f]} for f in inds}
    else: #check weight length - fit to eng
        for f in inds:
            for sk in inds[f]:
                #print(f, sk, len(inds[f][sk]), weights[f][sk].size) #for testing - eventual atoms.traj fails
                assert len(dats[f]['epot']) == weights[f][sk].size
    
    # presorting: base-keys
    bkeys = list(set(['_'.join(k.split('_')[:-1]) for k in dats]))
    # screen single states
    dat = {key:{} for key in bkeys}
    for key in bkeys:
        # keys of different runs
        skeys = [k for k in dats if '_'.join(k.split('_')[:-1]) == key]
        # keys for site selection
        sinds = {k:inds[k] for k in skeys}
        siteks = np.unique([kk for k in sinds for kk in sinds[k].keys()])
        # create data structure
        dat[key].update({sitek:{'epot':{},'etot':{}} for sitek in siteks})
        for sk in siteks:
            for etype in ['epot','etot']:
                # raw MD data - throw away first 5ps!
                ind = {k:np.array(sinds[k][sk],dtype=int)[np.where(np.array(sinds[k][sk]) > icut)[0]] for k in sinds if sk in sinds[k]}
                wts = {k:weights[k][sk] for k in sinds if sk in sinds[k]} #weights
                m_runs = np.array([np.mean(dats[k][etype][ind[k]]*wts[k][ind[k]]) for k in skeys if k in ind and ind[k].size > 0])
                std_runs = np.array([np.std(dats[k][etype][ind[k]]*wts[k][ind[k]]) for k in skeys if k in ind and ind[k].size > 0])
                if m_runs.size > 0: #non-evaluated site (shares)
                     # average state data
                     m_conf = np.mean(m_runs)
                     # error via thermal fluctuation
                     err_therm = np.mean(std_runs)
                     # error of total energy fluctuation
                     err_total = np.std(np.hstack(tuple([dats[k][etype][ind[k]] for k in skeys if k in ind and ind[k].size > 0])))
                     # error based on configurations
                     err_conf = np.std(m_runs) / np.sqrt(len(m_runs)) # division due to assuming normal distribution
                 
                     dat[key][sk][etype].update({'mRuns':m_conf, 'errRuns':err_conf, 'errtherm':err_therm, 'errtotal':err_total})
                else:
                    del dat[key][sk]
                    break
    return(dat)

def make_refs(dat,refs):
    kB = 8.6173303e-5; thkbt = (3./2.)*kB*300.
    dref = {}
    for rtag in refs:
        # potential energy
        epot = np.sum([dat['Epot']['E%s'%e] for e in rtag])
        # kinetic energy -> nCO+mH2+kO2
        ekin = 0; etrs = 0
        es = [e for e in rtag]
        cs = [e for e in rtag if e == 'C']
        for c in cs:
            es.remove('O')
            ekin += dat['Ekin']['CO'].mean()
            etrs += thkbt
        for a in ['H','O']:
            aa= [e for e in es if e == a]
            ekin += len(aa)*dat['Ekin']['%s2'%a].mean()/2.
            etrs += 0.5*thkbt

        dref.update({rtag:{'epot':epot,'etot':epot+ekin+etrs}})
    return(dref)

def eval_aimd_rel_energies2(dats,inds=None,weights=None,icut=5000):
    #TODO: replace eval_aimd_rel_energies
    #TODO: adjust Cu111 (ref_01, CO_01), same for Pt - add new calcs!!!, ctypes is actually NOT enough!!
    dat = get_energies()
    aref = make_refs(dat,['OCCHO', 'CHO', 'CO', 'COH', 'OH', 'OOH'])

   # TODO: delete if really not needed
   #ctypes = {'Cu211_15H2O_OCCHO':["Cu211_15H2O_0%i","Cu211_15H2O_OCCHO_0%i",range(1,5),[1,3,4,5]],\
   #         'Cu211_15H2O_Na+_OCCHO':["Cu211_15H2O_Na+_0%i","Cu211_15H2O_Na+_OCCHO_0%i",range(1,5),range(1,5)],\
   #         'Cu211-3x4_21H2O_Na+_OCCHO':["Cu211-3x4_21H2O_Na+_0%i","Cu211-3x4_21H2O_Na+_OCCHO_0%i",range(1,5),range(1,5)],\
   #         'Cu211_15H2O_CHO':["Cu211_15H2O_0%i","Cu211_15H2O_CHO_0%i",range(1,5),range(1,5)],\
   #         'Cu211_15H2O_COH':["Cu211_15H2O_0%i","Cu211_15H2O_COH_0%i",range(1,5),[1,2,3,5]],\
   #         'Cu211_15H2O_CO':["Cu211_15H2O_0%i","Cu211_15H2O_CO_0%i",range(1,5),range(1,5)],\
   #         'Cu211_15H2O_OH':["Cu211_15H2O_0%i","Cu211_15H2O_OH_0%i",range(1,5),range(1,5)],\
   #         'Cu211-6x4_42H2O_Na+_OCCHO':["Cu211-6x4_42H2O_Na+_0%i","Cu211-6x4_42H2O_Na+_OCCHO_0%i",range(1,5),range(1,5)],\
   #         'Cu211-15H2O-0.25q_OCCHO':["Cu211_15H2O-0.25q_0%i","Cu211_15H2O-0.25q_OCCHO_0%i",range(1,5),[1,3,4,5]],\
   #         'Cu111_24H2O_CO':["Cu111_24H2O_0%i","Cu211_24H2O_CO_0%i",range(2,5),range(2,5)],\
   #         'Au111_24H2O_CO':["Au111_24H2O_0%i","Au211_24H2O_CO_0%i",range(1,5),range(1,5)],\
   #         'Pt111_24H2O_CO':["Pt111_24H2O_0%i","Pt211_24H2O_CO_0%i",range(2,6),range(2,5)],\
   #         'Cu111_24H2O_OH':["Cu111_24H2O_0%i","Cu211_24H2O_OH_0%i",range(2,5),range(1,5)],\
   #         'Au111_24H2O_OH':["Au111_24H2O_0%i","Au211_24H2O_OH_0%i",range(1,5),range(1,5)],\
   #         'Pt111_24H2O_OH':["Pt111_24H2O_0%i","Pt211_24H2O_OH_0%i",range(2,6),range(1,5)],\
   #         }

    # compute single states ref, ads(configurational error (Delta-mean), thermal deviation)
    dat = _post_sort_process_aimd_data(dats,inds,weights,icut)

    d_ads = {k:dat[k] for k in dat if np.any([k.find(m) != -1 for m in ['OCCHO','CHO','COH','CO','OH','OOH']])}
    d_ref = {k:dat[k] for k in dat if np.all([k.find(m) == -1 for m in ['OCCHO','CHO','COH','CO','OH','OOH']])}

    reldat = __compute_aimd_rel_energies(d_ads, d_ref, aref)
    return(reldat)

def eval_aimd_solvation(dats, inds=None, weights=None):
    dat = get_energies()
    aref = make_refs(dat,['OCCHO', 'CHO', 'CO', 'COH', 'OH']) # Needs to get changed to H2O
    #aref = {k:{'etot':0.0,'epot':0.0} for k in ['OCCHO', 'CHO', 'CO', 'COH', 'OH']} # TODO: neater when changed to H2O ref here and no H2O ref to be computed in initial evaluation
    
    dat = _post_sort_process_aimd_data(dats, inds, weights, icut=1) # icut is indeces - only throw out first one
    
    d_ads = {k:dat[k] for k in dat if np.any([k.find(m) != -1 for m in ['OCCHO','CHO','COH','CO','OH']])}
    d_ref = {k:dat[k] for k in dat if np.all([k.find(m) == -1 for m in ['OCCHO','CHO','COH','CO','OH']])}

    reldat = __compute_aimd_rel_energies(d_ads, d_ref, aref)

    return(reldat)

def __compute_aimd_rel_energies(d_ads, d_ref, aref):
    reldat = {kads:{} for kads in d_ads}
    for kads in d_ads:
        for site in d_ads[kads]:
            reldat[kads].update({site:{'epot':{},'etot':{}}})
            for etype in ['epot','etot']:
                kref = [k for k in d_ref if k == '_'.join(kads.split('_')[:-1])][0]
                Erel = d_ads[kads][site][etype]['mRuns'] - d_ref[kref]['full'][etype]['mRuns'] - aref[kads.split('_')[-1]][etype]
                Eerr = np.linalg.norm([d_ads[kads][site][etype]['errRuns'], d_ref[kref]['full'][etype]['errRuns']])
                Terr = np.linalg.norm([d_ads[kads][site][etype]['errtherm'], d_ref[kref]['full'][etype]['errtherm']])
                terr = np.linalg.norm([d_ads[kads][site][etype]['errtotal'], d_ref[kref]['full'][etype]['errtotal']])
                reldat[kads][site][etype].update({'Erel':Erel,'Eerr':Eerr,'errtherm':Terr,'errtotal':terr})
    return(reldat)

def eval_aimd_rel_energies_ions(dats,inds):
    ads_p = ['OCCHO', 'CHO', 'CO', 'COH', 'OH']
    #aref = {'Li+': {'epot': -1.96907424, 'etot': 0.0}, 'Na+': {'epot': -1.45347561, 'etot': 0.0}, 'K+': {'epot': -1.08927585, 'etot': 0.0}} #wrong FCC
    aref = {'Li+': {'epot': -3.94282342/2.0, 'etot': 0.0}, 'Na+': {'epot': -2.90921586/2.0, 'etot': 0.0}, 'K+': {'epot': -1.08927585, 'etot': 0.0}}
    
    # compute single states ref, ads(configurational error (Delta-mean), thermal deviation)
    dat = _post_sort_process_aimd_data(dats,inds)

    d_ads = {k:dat[k] for k in dat if np.any([k.find(m) != -1 for m in ['Na+','Li+','K+']])}
    d_ref = {k:dat[k] for k in dat if np.all([k.find(m) == -1 for m in ads_p+['Na+','Li+','K+']])}

    del_keys = [k for k in d_ads if np.any([k.find(m) != -1 for m in ads_p+['3x4']])]
    for dk in del_keys:
        del(d_ads[dk])

    reldat = __compute_aimd_rel_energies(d_ads, d_ref, aref)
    return(reldat)
            
def _sample_density(folder,tstart):
    ions = []
    if folder.find("Na+") != -1:
        ions = [11]
    elif folder.find("K+") != -1:
        ions = [19]
    elif folder.find("Li+") != -1:
        ions = [3]
    # add slab to ions - exact distance determination
    ions.append(chemical_symbols.index(folder.split('_')[0].split('-')[0][:-3]))
    binc, hist_dicts = _get_density(folder,ions,tstart=tequ[folder])
    return(binc, hist_dicts)

def _load_wf_dat(folders, bpath=None):
    # quick load wfs
    if bpath == None:
        bpath = '/home/cat/heenen/Workspace/benchmarking_calcs/Cu_AIMD_runs/runs_AIMD'
    fpkl = bpath+'/output/aimd_wf_dat.pkl'; wfs = {}
    if os.path.isfile(fpkl):
        wfs = _load_pickle_file(fpkl)
    for f in folders:
        if f in wfs and len(wfs[f]) == 0:
            del wfs[f]
            _write_pickle_file(fpkl,wfs)
        if f not in wfs:
            wfdat, wf_approx = get_run_aimd_wf(bpath+'/'+f)
            if len(wfdat) > 0:
                wfs.update({f:wfdat})
                _write_pickle_file(fpkl,wfs)
                print('reading workfunction: %s'%f)
    return(wfs)

def _format_wf_data(wfs, tstart=0, tstop=-1):
    # adjust tstop
    tstop0 = tstop
    if tstop == -1:
        tstop0 = 1e9 # should always guarantee full traj
    # pre-sort wfs
    fkeys = list(set(['_'.join(k.split('_')[:-1]) for k in wfs]))
    wf_f = {fk:{} for fk in fkeys}
    for f in wfs:
        fbase = '_'.join(f.split('_')[:-1])
        wfdat = wfs[f]
        wfdat = [wfdat[t] for t in wfdat if t >= tstart and t < tstop0]
        #wf_f[fbase] = wf_f[fbase] + wfdat
        wf_f[fbase].update({f:wfdat})
    return(wf_f)
    
def _make_wf_table(outfile, folders, tstart=0, tstop=-1):
    wfs = _load_wf_dat(folders)
    wf_f = _format_wf_data(wfs,tstart, tstop)

    wf_f = {fk:[np.mean(wf_f[fk][f]) for f in wf_f[fk]] for fk in wf_f}
    wf_f = {fk:[np.mean(wf_f[fk]), np.std(wf_f[fk])] for fk in wf_f}
    ##wf_f = {fk:[np.mean(wf_f[fk]), np.std(wf_f[fk])] for fk in wf_f}

    header = '      AIMD-system         &           WF (eV)     \\\\ \n'
    for w in wf_f:
        header += ' %s  &   %.3f $\pm$ %.3f \\\\ \n'%(w, wf_f[w][0], wf_f[w][1])

    with open(outfile, 'w') as f:
        f.write(header)
    
def _process_energies(filename, edats, **kwargs): 
    ekey = 'etot'
    # last 5ps: -30ps:; all Edrift --> linear fit
    # make data for table: mean energy and std; plot with running averages stacked per surface
    ads = ['OCCHO', 'CHO', 'CO', 'COH', 'OH', 'OOH']
    gref = get_energies(); aref = make_refs(gref,ads)
    aref.update({'clean':{ekey:0.0}})

    # prep data
    fs = _sort_sub_runs(list(edats.keys()))
    fs = {k:fs[k] for k in fs if np.all([k.find(m) == -1 for m in ['Li+','Na+','K+','q']])}
    
    # clean_refs
    tstart = 5000
    if tstart in kwargs:
        tstart = kwargs['tstart']
    refk = [f for f in fs if len(f.split('_')) == 2]
    ref = {f:np.mean([np.mean(edats[sk][ekey][tstart:]) for sk in fs[f]]) for f in refk}
    refstd = {f:np.std([np.mean(edats[sk][ekey][tstart:]) for sk in fs[f]]) for f in refk}
    
    Erav = {f:[] for f in fs}; out_str = ""
    for f in fs:
        edt = []; rdt = []
        for sub in fs[f]:
            eng = edats[sub][ekey][tstart:]
            popt, f_lin = get_linear_fit(np.arange(0,eng.size)/1000.,eng)
            edt.append(np.mean(eng)-ref['_'.join(f.split('_')[:2])]-aref[__get_ads(f)][ekey]); rdt.append(popt[1]) #for table
            Erav[f].append(_running_av(eng)[::100] - aref[__get_ads(f)][ekey]) # every 0.1 ps
        print(f, np.mean(edt), np.linalg.norm([np.std(edt), refstd['_'.join(f.split('_')[:2])]]), np.std(edt), refstd['_'.join(f.split('_')[:2])])
        out_str += "%s"%f + ''.join([" & %.2f"%e for e in edt]) + \
            ''.join([" & %.0f"%(r*1000.) for r in rdt]) + ' \\\\ \n'
    with open('output/%s.txt'%filename,'w') as outfile:
        outfile.write(out_str)
    
    _plot_running_av_overview(filename, Erav, **kwargs)

def __get_ads(f):
    if len(f.split('_')) == 2:
        return('clean')
    else:
        return(f.split('_')[2])

if __name__ == "__main__":

    #### Cu211
    folders = ["Cu211_15H2O%s_0%i"%(a,i) for a in ['','_CO','_CHO','_OH'] for i in range(1,5)]
    folders += ["Cu211_15H2O%s_0%i"%(a,i) for a in ['_OCCHO'] for i in [1,3,4,5]]
    folders += ["Cu211_15H2O%s_0%i"%(a,i) for a in ['_COH'] for i in [1,2,3,5]]
    #### ions and charges
    folders += ['Cu211_15H2O_%s_0%i'%(a,i) for a in ['Na+_OCCHO','Na+','Li+','K+'] for i in range(1,5)]
    folders += ["Cu211-3x4_21H2O_%s_0%i"%(a,i) for a in ['Na+_OCCHO','Na+'] for i in range(1,5)]
    folders += ["Cu211-6x4_42H2O_%s_0%i"%(a,i) for a in ['Na+_OCCHO','Na+'] for i in range(1,5)]
    folders += ["Cu211_15H2O-0.25q_0%i"%(i) for i in range(1,5)]
    folders += ["Cu211_15H2O-0.25q_OCCHO_0%i"%(i) for i in [1,3,4,5]]
    #### runs Cu/Au/Pt 111
    folders += ['%s111_24H2O%s_0%i'%(e,a,i) for e in ['Cu','Au','Pt'] \
                            for a in ['','_CO','_OH'] for i in range(1,5)]
    folders += ['%s111_24H2O%s_05'%(e,a) for e in ['Cu','Pt'] for a in ['','_CO']]
    [folders.remove(k) for k in ['%s111_24H2O%s_01'%(e,a) for e in ['Cu','Pt'] for a in ['','_CO']]]
    #folders = []
    folders += ['Au111_24H2O_OOH_0%i'%i for i in range(1,5)]
    folders += ['Pt111_24H2O_OOH_0%i'%i for i in range(1,5)]
    
    # large benchmark tests
    Lfolders = ['Au111_48H2O_0%i'%i for i in [1,2,3,4]]
    Lfolders += ['Au111_48H2O_OH_0%i'%i for i in [1,2,3]]#,4]]
    #Lfolders += ['Pt111-6x4_48H2O_01', 'Pt111-6x4_48H2O_OH_01']
    Lfolders += ['Pt111_48H2O_02', 'Pt111_48H2O_03', 'Pt111_48H2O_OH_01', 'Pt111_48H2O_OH_02']
    #Lfolders += ['Pt111_48H2O_02', 'Pt111_48H2O_03']
    folders = Lfolders
    
    #folders = ['Pd111_24H2O_01']
        
    #### To compute H2O gas ref for solvation energy - (in principle not necessary)
    gref = get_energies()
    h2o_ref = gref['Epot']['EH']*2+gref['Epot']['EO']

    #### pre-equilibrium time 5ps for plotting
    ttequ = len(folders)*[5e3]
    #ttequ = len(folders)*[1e3]
    tequ = {folders[i]:ttequ[i] for i in range(0,len(folders))}

 #  #### energy conservation thingies:
    fpkl = '../reoptimize_RPBE-D3/output/aimd_engs_dat.pkl'
 # #leave_out = ['Pt111_24H2O_05'] + ["Cu211_15H2O-0.25q_0%i"%(i) for i in range(1,5)] + ['Cu211_15H2O_OCCHO_01'] + ["Cu211_15H2O-0.25q_OCCHO_0%i"%(i) for i in range(1,6)]
    if os.path.isfile(fpkl):
        edats = _load_pickle_file(fpkl)
 ###for f in folders:
 ###    if f not in edats and f not in leave_out:
 ###        edats.update({f:get_vasp_energetics(f)})
 ###        _write_pickle_file(fpkl, edats)
 # #for f in leave_out: # ONLY to correct edats according to post_reopt delecte!
 # #    if f in edats:
 # #        print("removing %s"%f)
 # #        del edats[f]
 # #        _write_pickle_file(fpkl, edats)
 ###assert False
   #edats = {f:edats[f] for f in edats if f.split('_')[2] != 'OOH'}
   #ecorr = {'Pt111_24H2O_OOH_03':range(0,20000), 'Pt111_24H2O_OOH_04':range(0,10000)}
   #for f in ecorr:
   #    edats[f] = {k:edats[f][k][ecorr[f]] for k in edats[f] if ecorr[f][-1] < len(edats[f][k])}
   #_process_energies('overview_Eav_drift', edats, ylim=(-3.0,4.8), fheight=3.37, lrbt=[0.1,0.90,0.15,0.98])
   #quit()
##############################################################
   #fpkl = 'output/aimd_engs_large.pkl'
   #if True: #not os.path.isfile(fpkl):
   #    edatsL = {f:get_vasp_energetics(f) for f in Lfolders}
   #    _write_pickle_file(fpkl, edatsL)
   #edatsL = _load_pickle_file(fpkl)
   #_process_energies('overview_Eav_drift_large', edatsL, tstart=8000, ylim=(-1.5,2.0), \
   #    surftag={'Pt111':r'Pt(111)-48H$_2$O', 'Au111':r'Au(111)-48H$_2$O'}, \
   #    #surftag={'Pt111':r'Pt(111)-6$\times$4', 'Au111':r'Au(111)-48H$_2$O'}, \
   #    fwidth=3.37, lrbt=[0.16,0.97,0.25,0.97],legbb=None,legloc=4)
   #quit()
##############################################################
   #_make_wf_table('output/wf_table.txt',folders,tstart=5e3)
   #_process_densities('z_density_comparison', folders)#, write_eps=True)
   #quit()
    
   #pklfile = 'output/h2o_adsorption_plot.pkl'
   #plotout = _load_pickle_file(pklfile)
   #_plot_h2o_adsorption('h2o_adsorption', plotout, write_eps=True)
   #quit()
    
   #flds = _sort_sub_runs(folders)
   #surfs = {s:flds[s] for s in ['Pt111_24H2O', 'Cu211_15H2O', 'Cu111_24H2O', 'Au111_24H2O']}
   ##surfs = {s:flds[s] for s in ['Pt111_48H2O']}
   #for surf in surfs:
   #    sdens = _av_dens(surfs[surf])
   #    plot_density('density_av_%s'%surf, sdens['binc'], {k:sdens[k] for k in sdens if k != 'binc'}, dens=False)
   #flds = _sort_sub_runs(folders)
   #plot_density_fancy('z_density_comparison_Pt111_48H2O', flds['Pt111_48H2O'])
   #quit()
    
    d = _load_pickle_file("output/Esol.pkl"); dats_sol = d['d_psol']; dats = d['dats']
    for folder in folders:
        dat = get_vasp_energetics(folder)
        if not os.path.isfile('output/'+'EvsT_av_%s.pdf'%folder) or \
            _get_time_latest_vaspout(folder) > os.path.getmtime('output/'+"EvsT_av_%s.pdf"%folder):
            plot_engs_vs_time("EvsT_%s"%(folder),dat,tstart=tequ[folder])
            plot_av_engs_vs_time("EvsT_av_%s"%(folder),dat,tstart=tequ[folder])

        dats.update({folder:dat})

     ## print(folder)
     ## d_psol = get_run_dvac_sp(folder, dat, h2o_ref)
     ## dats_sol.update({folder:d_psol})

       #wfdat, wf_approx = get_run_aimd_wf(folder)
       #if len(wfdat) > 0:
       #    plot_wf_vs_time("WF_%s"%(folder),wfdat,wf_approx)

####    # any of the expensive stuff where one needs to read atoms
####    if not os.path.isfile('output/'+"density_%s.pdf"%folder) or \
####        _get_time_latest_vaspout(folder) > os.path.getmtime('output/'+"density_%s.pdf"%folder):
####        binc, hist_dicts = _sample_density(folder,tstart=tequ[folder])
####        plot_density("density_%s"%folder,binc,hist_dicts)
####    
####    denspkl = '%s/density_dat.pkl'%folder
####    if not os.path.isfile(denspkl):
####        binc, hist_dicts = _sample_density(folder,tstart=tequ[folder])
####        _write_pickle_file(denspkl, {'binc':binc, 'hist_dicts':hist_dicts})

   ## safe energy data
   #_write_pickle_file("output/Esol.pkl", {'dats':dats, 'd_psol':dats_sol}) # - for plotting routines
   #plot_esol_vs_time('dEsol_overview', dats_sol)

    # plot convergence of adsorption energy

   #plot_av_dE_vs_time("dEvsT_av_15H2O_OCCHO",dats["Cu211_15H2O_OCCHO_01"],dats["Cu211_15H2O_01"],refOCCHO,tstart=10e3)
   #plot_av_dE_vs_time("dEvsT_av_15H2O_Na+_OCCHO",dats["Cu211_15H2O_Na+_OCCHO_01"],dats["Cu211_15H2O_Na+_01"],refOCCHO,tstart=4e3)

    #TODO: this should be a function! so can be assessed by post_reoptimized
   #eval_aimd_rel_energies(dats)

