#!/usr/bin/env python

from shutil import copyfile
import numpy as np
from ase.io import read, write
from ase.data import chemical_symbols as chemsym
from ase.build import sort
from cathelpers.slurmsubmitter import submit_jobs_slurm
from cathelpers.misc import _load_pickle_file
from cathelpers.atom_manipulators import _return_atoms_by_ind
import os, sys
from copy import deepcopy

sys.path.append("/home/cat/heenen/Workspace/benchmarking_calcs/Cu_AIMD_runs/scripts_vasp")
from post_process_traj import _read_restart_atoms, check_vasp_opt_converged

sys.path.append('/home/cat/heenen/Workspace/benchmarking_calcs/Cu_AIMD_runs/convergence_tests')
from test_convergence import _prep_potcar
        
from test_site_occupation import _arr2hash

def _select_solvent_inds(wpath,ind_sp,atno,no_fix):
    # select winds for snapshots
    # tracking of adsorbate: 
    # if different adsorbate (i.e. None - OH) take winds from step before
    # if adsorbate ads change covered in special cases (i.e. COH)
    
    atag = ''; satag = ''#tag to figure out adsorbate formula
    if len(wpath.split('.')[-1].split('_')) == 4:
        atag = wpath.split('.')[-1].split('_')[-2]
        satag = ''.join(np.sort([a for a in atag]))

    # compensate bad starting structure 111-surfaces and others???
    corr_inds = {'%s111_24H2O_%s_0%i'%(e,a,i):[48,73] for e in ['Cu','Au','Pt'] for a in ['CO','OH'] for i in range(1,5)}
    corr_inds.update({'Au111_24H2O_OH_0%i'%i:[31,64] for i in [2,3]}) #apparently changed 
    corr_inds.update({'Pt111_24H2O_OH_0%i'%i:[7,52] for i in [1]}) #apparently changed 
    corr_inds.update({'%s111_24H2O_0%i'%(e,i):[] for e in ['Cu','Au','Pt'] for i in range(1,5)})

    # iterate winds
    site_dat = _load_pickle_file(wpath+'/site_sampling.pkl'); winds = []; ainds = []
    
    # for testing:
    if ind_sp == None:
        ind_sp = range(0,sum([len(site_dat[a]['traj_inds']) \
                                    for a in range(len(site_dat))]),100)
    
    elads = {}; whshs = {}
    for i in ind_sp:
        ii = [j for j in range(len(site_dat)) if i in site_dat[j]['traj_inds']][0]
        wi = site_dat[ii]['water_inds']; ai = site_dat[ii]['ads_inds']
        af = ''.join(np.sort([chemsym[atno[c]] for c in ai]))
        if i == 0:
            pre = wi
            if wpath.split('/')[-1] in corr_inds:
                aitemp = corr_inds[wpath.split('/')[-1]]
                wi = np.array([tm for tm in np.hstack((wi,ai)) if tm not in aitemp],dtype=int)
                pre = wi
        else:
            if af != satag: # case strong interaction
                #special case COH deprotonation
                if not (atag == 'COH' and af == 'CO'):
                    wi = pre
            else:
                pre = wi
        winds.append(wi)
        
        # for checking:
        ads = ''.join(np.sort([chemsym[atno[t]] for t in range(len(atno)) if (atno[t] != no_fix and t not in wi)]))
        hshs = _arr2hash(wi)
        if ads not in elads:
            elads.update({ads:0})
        if hshs not in whshs:
            whshs.update({hshs:0})
        elads[ads] += 1; whshs[hshs] += 1
    
    # Exception if adsorbate inds unexpected
    if np.any([kads != satag for kads in elads]):
        print(elads)
        print(whshs)
        raise Exception('something goes wrong in ads determination')
    return(winds)

def __remove_water(a, water_ind):
    a = deepcopy(a)
    del a[water_ind]
  ########## for checking: #####
  # atno = a.get_atomic_numbers()
  # print(atno)
  # print({t:np.where(atno == t)[0].size for t in np.unique(atno)})
  # print(a.constraints[0].index)
  # ######## for checking: #####
    return(a)


def prep_sp_no_solvent(wpath):
    # prepare sp path
    sppath = '%s/singlepoints_nosolvent'%wpath
    print('in %s preparing %s'%(wpath,sppath))
    templ_path = '/home/cat/heenen/Workspace/benchmarking_calcs/Cu_AIMD_runs/template_files_VASP'
    if not os.path.isdir(sppath):
        os.mkdir(sppath)

    # read-in atoms
    traj = _read_restart_atoms(wpath)
    ind_sp = range(0,len(traj),100) # every 0.1 ps

    atno = traj[0].get_atomic_numbers()
    no_fix = np.unique(traj[0].get_atomic_numbers()\
        [traj[0].constraints[0].index])[0]
    #selects for each ind_sp
    winds = _select_solvent_inds(wpath,ind_sp=ind_sp,atno=atno,no_fix=no_fix) 
    
    rdirs = []; iw = 0
    for i in ind_sp:
        a = __remove_water(traj[i], winds[iw]); iw += 1
        a = sort(a, tags=a.get_atomic_numbers()) #resorting needed
        cdir = sppath+'/sp_step{0:05d}'.format(i)
        if not os.path.isdir(cdir):
            os.mkdir(cdir)
        if not os.path.isfile(cdir+"/OUTCAR") and not os.path.isfile(cdir+"OUTCAR.gz") or \
            not check_vasp_opt_converged(cdir):
            write(cdir+"/POSCAR",a)
            _prep_potcar(cdir)
            copyfile(wpath+"/KPOINTS",cdir+"/KPOINTS")
            copyfile(templ_path+"/INCAR_sp_benchmark",cdir+"/INCAR")
            rdirs.append(cdir.split('/')[-1])
    _write_chain_submit_file(sppath, rdirs, nbatch=100)

def prep_sp_wf(sppath):
    print('in %s'%sppath)
    templ_path = '/home/cat/heenen/Workspace/benchmarking_calcs/Cu_AIMD_runs/template_files_VASP'
    if not os.path.isdir(sppath):
        os.mkdir(sppath)
    wpath = '/'.join(sppath.split('/')[:-1])
    traj = _read_restart_atoms(wpath)
    #ind_sp = range(0,len(traj),1000) # every 0.5 ps
    ind_sp = range(0,len(traj),500) # every 0.5 ps
    cdirs = []; rdirs = []
    for i in ind_sp:
        cdir = sppath+'/sp_step{0:05d}'.format(i)
        if not os.path.isdir(cdir):
            os.mkdir(cdir)
        if (not os.path.isfile(cdir+"/OUTCAR") and not os.path.isfile(cdir+"/OUTCAR.gz")) or \
            not check_vasp_opt_converged(cdir):
            write(cdir+"/POSCAR",traj[i])
            copyfile(wpath+"/POTCAR",cdir+"/POTCAR")
            copyfile(wpath+"/KPOINTS",cdir+"/KPOINTS")
            copyfile(templ_path+"/INCAR_WFsp_benchmark",cdir+"/INCAR")
            copyfile(templ_path+"/submit_vasp_WFsp.slurm",cdir+"/submit_vasp_WFsp.slurm")
            cdirs.append(cdir)
            rdirs.append(cdir.split('/')[-1])
    _write_chain_submit_file(sppath, rdirs, nbatch=100, submit_f_name='submit_vasp_WFsp_chain')
    #submit_jobs_slurm(cdirs, f_submit='submit_vasp_WFsp.slurm')

def _write_chain_submit_file(wpath, cdirs, nbatch, \
                                        submit_f_name='submit_vasp_sp_chain'):
    # file header
    hdr = "#!/bin/bash\n#SBATCH -J job\n#SBATCH -p xeon16\n"
    hdr += "#SBATCH -N 1\n#SBATCH --ntasks-per-node=16\n"
    hdr += "#SBATCH -t 20:00:00\n#SBATCH -o slurm.%j.out\n" #8 hrs roughly for 100
    hdr += "#SBATCH -e err\n\nmodule purge\nmodule load intel\n"
    hdr += "export PATH=/home/cat/zwang/bin/cattheory/vasp/5.4.4:$PATH\n\n"
    hdr += "cwd=$(pwd)\n\n"

    mdl = "\ncd %s\nmpirun vasp_std\ncd $cwd\n\n"

    # make nbatch - job files
    nb = list(range(0,len(cdirs),nbatch)) + [len(cdirs)]
    for i in range(len(nb) - 1):
        indirs = cdirs[nb[i]:nb[i+1]]
        ftxt = hdr
        for indir in indirs:
            ftxt += mdl%indir
        with open(wpath+'/{0}_{1:02d}.slurm'.format(submit_f_name,i),'w') as sf:
            sf.write(ftxt)


if __name__ == "__main__":
    
    folders = [f for f in os.listdir('.') if f.find("Cu211_15H2O_") != -1]
   #folders = ["Cu211_15H2O_0%i"%i for i in range(1,5)]
   #folders = ["Cu211_15H2O_CHO_0%i"%i for i in range(1,5)] + \
   #          ["Cu211_15H2O_CO_0%i"%i for i in range(1,5)] + \
   #folders = ["Cu211_15H2O_OCCHO_0%i"%i for i in [1,3,4,5]]
   #folders = ["Cu211_15H2O_COH_0%i"%i for i in range(1,4)]
   #folders += ["Cu211_15H2O_OH_0%i"%i for i in range(1,5)]
   #           ["Cu211-3x4_21H2O_Na+_0%i"%i for i in range(1,5)] + \
   #           ["Cu211-3x4_21H2O_Na+_OCCHO_0%i"%i for i in range(1,5)] +\
   #           ["Cu211_15H2O_K+_0%i"%i for i in range(1,5)]
   #folders = ["Cu211_15H2O_Li+_0%i"%i for i in range(1,5)]
              #["Cu211-3x4_21H2O_Na+_OCCHO_0%i"%i for i in range(1,5)] 
    
   ## new runs - singlepoint WF - submitted
  ##folders = ["Cu111_24H2O_0%i"%i for i in range(1,5)]
  ##folders = ["Cu111_24H2O_CO_0%i"%i for i in range(1,5)]# +\
  ##folders = ["Cu111_24H2O_OH_0%i"%i for i in range(1,5)]
  ##folders = ["Au111_24H2O_0%i"%i for i in range(1,5)]
  ##folders = ["Au111_24H2O_CO_0%i"%i for i in range(1,5)]
  ##folders += ["Au111_24H2O_OH_0%i"%i for i in range(1,5)]
  ##folders = ["Pt111_24H2O_CO_0%i"%i for i in range(1,5)]
  ##folders = ["Pt111_24H2O_0%i"%i for i in range(1,5)]
  ##folders += ["Pt111_24H2O_OH_0%i"%i for i in range(1,5)]
  ##folders += ["Cu211_15H2O_COH_0%i"%i for i in [5]]
  ##folders += ["Cu211_15H2O_COH_0%i"%i for i in range(1,4)]
   #folders += ["Cu211-6x4_42H2O_Na+_0%i"%i for i in range(1,5)]
   #folders += ["Cu211-6x4_42H2O_Na+_OCCHO_0%i"%i for i in range(1,5)]
   ## next best batch - sinlgepoint WF
    #folders = ["Cu211_15H2O-0.25q_0%i"%i for i in range(1,5)]
    #folders = ["Cu211_15H2O-0.25q_OCCHO_0%i"%i for i in [1,3,4,5]]
    #folders = ['Cu211_15H2O_%s_0%i'%(a,i) for a in ['Na+_OCCHO','Na+'] for i in range(1,5)]
#   recheck these
#   folders = ['Cu211-6x4_42H2O_Na+_OCCHO_0%i'%(i) for i in range(3,5)]
#   folders += ['Cu211-6x4_42H2O_Na+_0%i'%(i) for i in range(4,5)]
#   folders = ['Cu211-6x4_42H2O_%s_0%i'%(a,i) for a in ['Na+_OCCHO','Na+'] for i in range(1,5)]
#   folders += ["Cu111_24H2O_0%i"%i for i in range(3,6)]
#   folders += ["Cu111_24H2O_CO_0%i"%i for i in range(3,6)]
#   folders += ["Cu111_24H2O_OH_0%i"%i for i in range(3,5)]
#   folders += ["Au111_24H2O_OH_0%i"%i for i in range(1,5)]
#   folders += ["Pt111_24H2O_0%i"%i for i in range(1,5)]
#   folders += ["Pt111_24H2O_CO_0%i"%i for i in range(1,6)]
#   folders += ["Pt111_24H2O_OH_0%i"%i for i in range(1,5)]

  #### rerun whats missing 
  ##folders = ['Cu111_24H2O_0%i'%i for i in [3,4,5]]
  ##folders += ['Cu111_24H2O_CO_0%i'%i for i in range(1,5)]
  ##folders += ['Cu111_24H2O_OH_0%i'%i for i in range(1,4)]
  ##folders += ['Au111_24H2O_CO_0%i'%i for i in range(1,3)]
  ##folders += ['Au111_24H2O_OH_0%i'%i for i in range(1,5)]
  ##folders += ['Pt111_24H2O_0%i'%i for i in [1]]
  ##folders += ['Pt111_24H2O_CO_0%i'%i for i in [1]]
  ##folders += ['Pt111_24H2O_OH_0%i'%i for i in range(1,4)]
   #folders = ['Au111_48H2O_01', 'Au111_48H2O_OH_01']
   #folders = ['Au111_48H2O_02', 'Au111_48H2O_OH_02']
    #folders = ['Au111_48H2O_0%i'%i for i in [4]]
    folders = ['Pt111-6x4_48H2O_01']

    for f in folders:
        sppath = f+"/singlepoints_wf"
        prep_sp_wf(sppath)
        #prep_sp_no_solvent(f)


