#!/usr/bin/env python

import sys, os, gzip
import numpy as np
from copy import deepcopy
from os import popen
from time import time
from scipy.optimize import curve_fit

sys.path.append('/home/cat/heenen/Workspace/TOOLS/ASE-DTU/ase')
from ase.io import read, write
from ase.visualize import view
from ase.data import chemical_symbols as symbols
from ase.data.colors import jmol_colors
from ase.calculators import vasp

sys.path.append("/home/cat/heenen/Workspace/TOOLS/rtools")
from rtools.helpers.matplotlibhelpers import tumcolors as tumcs
import rtools.helpers.matplotlibhelpers as matplotlibhelpers
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.constants import golden_ratio, inch
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

#sys.path.append("../convergence_tests")
from transform_lattice_constant_standalone import identify_water
#from cathelpers.md_tools import atomic_diffusion_tools as mdtools
sys.path.append("/home/cat/heenen/Workspace/TOOLS/CatHelpers")
from cathelpers.atom_manipulators import find_max_empty_space
from cathelpers.misc import lines_key, _load_pickle_file, _write_pickle_file



#############################################################
################## NOTE: already worked in ##################
##############################################################
#def _set_plotting_env(width=3.37,height=3.37/ golden_ratio *1.5/2,\
#                    lrbt=[0.135,0.80,0.25,0.95],fsize=9.0,font='helvetica'):
#    # set plot geometry
#    rcParams['figure.figsize'] = (width, height) # x,y
#    rcParams['font.size'] = fsize
#    rcParams['figure.subplot.left'] = lrbt[0]  # the left side of the subplots of the figure
#    rcParams['figure.subplot.right'] = lrbt[1] #0.965 # the right side of the subplots of the figure
#    rcParams['figure.subplot.bottom'] = lrbt[2] # the bottom of the subplots of the figure
#    rcParams['figure.subplot.top'] = lrbt[3] # the bottom of the subplots of the figure
#
#    rcParams['xtick.top'] = True
#    rcParams['xtick.direction'] = 'in'
#    rcParams['ytick.right'] = True
#    rcParams['ytick.direction'] = 'in'
#
#    rcParams['legend.fancybox'] = False
#    #rcParams['legend.framealpha'] = 1.0
#    rcParams['legend.edgecolor'] = 'k'
#    #matplotlibhelpers.set_latex(rcParams,font=font) #poster

#def _get_val_OUTCAR(filename,string,i,j):
#    tstr = popen("grep '%s' %s"%(string,filename)).read().split()
#    val = [float(s) for s in tstr[i::j]]
#    return(np.array(val))
#
#def _read_xml_energetics(filename):
#    # NOTE xml parser probably better - but this quick'n'dirty works
#    tstr = popen("grep '%s' -B 4 %s"%("kinetic",filename)).read().split()
#    epot = np.array([float(s) for s in tstr[3::31]])
#    ekin = np.array([float(s) for s in tstr[15::31]])
#    etot = ekin+epot
#    return({'ekin':ekin,'epot':epot,'etot':etot,'looptime':np.zeros(ekin.size),'trajlen':np.array([ekin.size])})
#
#def _read_xml_singlepoint(filename):
#    tstr = popen("grep '%s' %s"%("e_fr_energy",filename)).readlines()
#    engs = [float(s.split()[2]) for s in tstr]
#    return({'epot':engs[-1]})
#
#def _read_OUTCAR_energetics(filename):
#    # Thermodynamics
#    ekin = _get_val_OUTCAR(filename,'kinetic energy EKIN   =',4,5)
#    epot = _get_val_OUTCAR(filename,'ion-electron   TOTEN  =',4,7)
#    etot = ekin+epot
#    # timeings
#    loop_time = np.mean(_get_val_OUTCAR(filename,'LOOP+',6,7))
#    return({'ekin':ekin,'epot':epot,'etot':etot,'looptime':loop_time,'trajlen':np.array([ekin.size])})
#
#def get_OUTCAR_energetics(wpath):
#    data = []
#    while os.path.isfile(wpath+"/OUTCAR"):
#        data.append(_read_OUTCAR_energetics(wpath+"/OUTCAR"))
#        wpath += '/restart'
#    dat = data[0]
#    for i in range(1,len(data)):
#        for key in dat:
#            dat[key] = np.hstack((dat[key],data[i][key]))
#    return(dat)
#
#def get_vasp_energetics(wpath):
#    data = []
#    while os.path.isfile(wpath+"/OUTCAR") or os.path.isfile(wpath+"/vasprun.xml"):
#        if os.path.isfile(wpath+"/OUTCAR"):
#            data.append(_read_OUTCAR_energetics(wpath+"/OUTCAR"))
#        elif os.path.isfile(wpath+"/vasprun.xml"):
#            data.append(_read_xml_energetics(wpath+"/vasprun.xml"))
#        wpath += '/restart'
#    dat = _join_data_list(data)
#    return(dat)
#
#def _join_data_list(data):
#    dat = data[0]
#    for i in range(1,len(data)):
#        for key in dat:
#            dat[key] = np.hstack((dat[key],data[i][key]))
#    return(dat)
    
#def plot_engs_vs_time(filename, dat, tstart):
#    label = {'ekin':r'E$_{\mathrm{kin}}$ (eV)',\
#            'epot':r'E$_{\mathrm{pot}}$ (eV)',\
#            'etot':r'E$_{\mathrm{tot}}$ (eV)',\
#            'x':r'simulation time (ps)'}
#    _plot_xe_vs_time(filename,dat,label,_nofunc,tstart=tstart)
#
#def _nofunc(indat):
#    return(indat)
#
#def plot_av_engs_vs_time(filename, dat, tstart):
#    label = {'ekin':r'$\langle$E$_{\mathrm{kin}} \rangle$ (eV)',\
#            'epot':r'$\langle$E$_{\mathrm{pot}} \rangle$ (eV)',\
#            'etot':r'E$_{\mathrm{tot}}$ (eV)',\
#            'x':r'simulation time (ps)'}
#    _plot_xe_vs_time(filename,dat,label,_running_av,tstart=tstart)
#
#def _running_av(indat):
#    y = np.cumsum(indat)/np.arange(1,indat.size+1)
#    return(y)
#
#def _plot_xe_vs_time(filename,dat,label,yfunc,ekeys=['ekin','epot','etot'],tstart=1e3):
#    _set_plotting_env(width=3.37,height=3.37,#/ golden_ratio *1.5/2,\
#                   lrbt=[0.25,0.95,0.15,0.95],fsize=9.0)
#    fig = plt.figure()
#    n = 0; axes = []
#    for e in ekeys:
#        ax = fig.add_subplot(len(ekeys)*100+11+n)
#        t = np.arange(tstart,dat[e].size)/1000.
#        y = yfunc(dat[e][int(tstart):])
#        ax.plot(t[::20],y[::20],color='k')
#        ax.set_ylabel(label[e])
#        axes.append(ax)
#        n += 1
#        # add line indicating AIMD traj restarts
#        if 'trajlen' in dat:
#            ttj = 0.0
#            for tt in dat['trajlen']:
#                ttj += tt
#                ax.axvline(x=ttj/1000.,lw=0.5,ls=':',color='k')
#    axes[-1].set_xlabel(label['x'])
#    [axes[a].set_xticklabels([]) for a in range(len(axes)-1)]
#    plt.subplots_adjust(hspace=0.0)
#    writefig(filename)
#   #matplotlibhelpers.write(filename,folder='output',\
#   #    write_info = False,write_png=False,write_pdf=True,write_eps=False)
#############################################################
#############################################################
#############################################################
    
def _plot_running_av_overview(filename, Erav, surftag = {}, tstart=5000, ylim=(-3.0,2.0), fwidth=2*3.37, fheight=3.37/ golden_ratio,lrbt=[0.1,0.90,0.25,0.98],legloc=1,legbb=(1.55,1.0)):
    _set_plotting_env(width=fwidth,height=fheight,\
                   lrbt=lrbt,fsize=9.0)

    csurf = {'Cu211':tumcs['tumred'], 'Cu111':tumcs['tumorange'],\
        'Au111':tumcs['acc_yellow'], 'Pt111':tumcs['darkgray']}
    clr = {'clean':'k', 'OH':tumcs['diag_red'], 'CO':tumcs['tumgreen'], \
        'CHO':tumcs['tumblue'], 'COH':tumcs['tumlightblue'], 'OCCHO':tumcs['diag_violet'],
        'OOH':tumcs['tumorange']}
    if len(surftag) == 0:
        surftag = {'Cu111': 'Cu(111)', 'Cu211':'Cu(211)', 'Au111':'Au(111)', 'Pt111':'Pt(111)'}

    fig = plt.figure()
    surfs = ['Cu211', 'Cu111', 'Au111', 'Pt111']; c = 0; axes = []
    surfs = [s for s in surfs if np.any([k.find(s) != -1 for k in Erav])]
    
    for surf in surfs:
        sf = [f for f in Erav if f.split('_')[0][:5] == surf]
        ax = fig.add_subplot(101+10*len(surfs)+c)
        ax.annotate(r'%s'%surftag[surf], xy=(0.15,0.65), xycoords='axes fraction',size=9, color=csurf[surf])
        refkey = [rk for rk in sf if len(rk.split('_')) == 2][0]
        ref = np.mean([e[-1] for e in Erav[refkey]])
        for f in sf:
            ads = __get_ads(f)
          #### hard-coded artefact dashed lines
          ##if ads == 'OOH' and surf == 'Pt111':
          ##    ind2 = int(200-tstart/100); ind3 = int(100-tstart/100)
          ##    ax.plot(np.arange(tstart/100.,tstart/100.+Erav[f][2].size)[ind2:]*0.1, Erav[f][2][ind2:]-ref, ls='--', color=clr[ads])
          ##    ax.plot(np.arange(tstart/100.,tstart/100.+Erav[f][3].size)[ind3:]*0.1, Erav[f][3][ind3:]-ref, ls='--', color=clr[ads])
          ##    Erav[f][2] = Erav[f][2][:ind2]; Erav[f][3] = Erav[f][3][:ind3]
          #### hard-coded artefact dashed lines
            for d in Erav[f]:
                t = np.arange(tstart/100.,tstart/100.+d.size)*0.1
                ax.plot(t, d-ref, ls='-', color=clr[ads])
        axes.append(ax)
        c += 1

    # legend
    dads = [__get_ads(f) for f in Erav]
    [axes[-1].plot(np.nan, np.nan, ls='-', color=clr[ads], label=r'%s'%ads) for ads in clr if ads in dads]
    axes[-1].legend(loc=legloc,prop={'size':7},framealpha=1.0,facecolor='w', bbox_to_anchor=legbb)

    axes[0].set_ylabel(r'$\langle \Delta E \mathrm{_{ad}^{total}} \rangle (t)$ (eV)')
    mdl = int((len(surfs)/2)-1)
    axes[mdl].set_xlabel(r'time (ps)')
    if len(axes)%2 == 0:
        axes[mdl].xaxis.set_label_coords(1.0, -0.18*((3.37/golden_ratio)/fheight))
    [ax.set_ylim(ylim) for ax in axes]
    [axes[a].set_yticklabels([]) for a in range(1,len(axes))]
    plt.subplots_adjust(wspace=0.05)

    writefig(filename)
            
def __get_ads(f):
    if len(f.split('_')) == 2:
        return('clean')
    else:
        return(f.split('_')[2])

#############################################################
################## NOTE: already worked in ##################
##############################################################

#def _prep_POSCAR_atoms(wpath):
#    if not os.path.isfile(wpath+"/atoms.traj") or \
#        os.path.getmtime(wpath+"/atoms.traj") < os.path.getmtime(wpath+"/vasprun.xml"):
#        atoms = read(wpath+"/vasprun.xml",':')
#        #atoms = read(wpath+"/OUTCAR",':')
#        write(wpath+"/atoms.traj",atoms)

#def _get_time_latest_OUTCAR(wpath):
#    while os.path.isfile(wpath+"/OUTCAR"):
#        tmp = os.path.getmtime(wpath+"/OUTCAR")
#        wpath += '/restart'
#    return(tmp)

#def _get_time_latest_vaspout(wpath):
#    while os.path.isfile(wpath+"/OUTCAR") or os.path.isfile(wpath+"/vasprun.xml"):
#        if os.path.isfile(wpath+"/OUTCAR"):
#            tmp = os.path.getmtime(wpath+"/OUTCAR")
#        else:
#            tmp = os.path.getmtime(wpath+"/vasprun.xml")
#        wpath += '/restart'
#    return(tmp)

#def _file_age_check(f_name, dt=30):
#    """ function to return whether file is older than dt """
#    if os.path.isfile(f_name):
#        tmp = os.path.getmtime(f_name)
#        t = (time() - tmp) / 60. # time in minutes
#        return(t > dt)
#    else:
#        return(False)

#def _read_restart_atoms(wpath):
#    _prep_POSCAR_atoms(wpath)
#    atoms = read(wpath+"/atoms.traj",':')
#    wpath += '/restart'
#    #while os.path.isfile(wpath+"/OUTCAR") and _file_age_check(wpath+"/OUTCAR",30):
#    while (os.path.isfile(wpath+"/vasprun.xml") and _file_age_check(wpath+"/vasprun.xml",30)) or os.path.isfile(wpath+"/atoms.traj"):
#        _prep_POSCAR_atoms(wpath)
#        a = read(wpath+'/atoms.traj',':')
#        atoms = atoms + a
#        wpath += '/restart'
#    return(atoms)

#def _get_density(wpath,atypes=[],tstart=0):
#    if type(wpath) == str: # if wpath str read atoms otherwise assumed as atoms
#        traj = _read_restart_atoms(wpath)
#    else:
#        traj = wpath
#    #traj = read(infile,':')
#    water = identify_water(traj[0])
#
#   ## sanity check for OCCHO adsorbate: nO - nwater = 2
#   #if np.where(traj[0].get_atomic_numbers() == 8)[0].size - len(water) != 2:
#   #    raise Exception('identification of water molecules went wrong')
#    # sanity check all waters have 2 hydrogens
#    for w in water:
#        assert len(water[w]) == 2
#
#    # indices of water and others
#    ind_Ow = list(water.keys())
#    ind_Hw = list(np.array(list(water.values())).flatten())
#    ind_o = [np.where(traj[0].get_atomic_numbers() == atype)[0] \
#                                            for atype in atypes]
#    inds = [ind_Ow,ind_Hw] + ind_o
#    
#    # cut traj here
#    traj = traj[int(tstart):]
#    # compute histogram
#    bins = np.linspace(0,traj[0].get_cell()[2,2],200) # hist by height
#    hists = [np.zeros(len(bins)-1) for i in range(len(inds))]
#    for snap in traj:
#        pos = snap.get_positions()
#        for i in range(0,len(inds)):
#            d = np.histogram(pos[inds[i],2],bins)
#            hists[i] += d[0]
#    for i in range(0,len(hists)): # normalize time
#        hists[i] /= len(traj) 
#
#    # convert histogram into density
#    dA = traj[0].get_volume()/traj[0].cell[2,2]
#    dh = bins[1] - bins[0]
#    f_convert = 1.660538921e-24/(dA*dh*(1e-8)**3) #conversion mol/cm3
#    for i in range(0,len(hists)):
#        hists[i] *= f_convert
#    
#    # make dict for plotting
#    dens_types = ['Ow','Hw']+[symbols[a] for a in atypes]
#    hist_dicts = {dens_types[i]:hists[i] for i in range(len(dens_types))}
#    
#    # bin center
#    binc = (bins[0:-1] + bins[1:])/2
#
#    return(binc,hist_dicts)

#def _plot_density(binc, hist_dicts, ax, dens=False):
#    colors = {'Ow':'r','Hw':'c'}
#    colors.update({el:jmol_colors[symbols.index(el)] \
#            for el in hist_dicts if el not in colors})
#    for el in hist_dicts:
#        nz = np.where(hist_dicts[el] != 0.0)[0]
#        if nz.size != 0:
#            #ax.plot(binc[nz]-binc[nz][0],hist_dicts[el][nz],ls='-',color=colors[el],label=r'%s'%el)
#            ax.plot(binc[nz]-binc[0],hist_dicts[el][nz],ls='-',color=colors[el],label=r'%s'%el)
#    hlim = np.array(ax.get_xlim()); d = hlim[1]-hlim[0]; hlim[0] += 0.2*d; hlim[1] -= 0.2*d
#    
#    if dens:
#        # water density
#        ax2 = ax.twinx()
#        ax2.hlines([1],*hlim,color='k',linestyle='--',linewidth=0.5)
#        nz = np.where(hist_dicts['Ow'] != 0.0)[0]
#        ax2.plot(binc[nz]-binc[nz][0],hist_dicts['Ow'][nz]*18,ls=':',color='k')
#        ax.plot(np.nan, np.nan, ls=':', c='k', label=r'$\rho_{\mathrm{H_2O}}$')
#    else:
#        ax2 = None
#        ax.hlines([1/18.],*hlim, color=colors['Ow'], linestyle='--', linewidth=0.5)
#        ax.hlines([2/18.],*hlim, color=colors['Hw'], linestyle='--', linewidth=0.5)
#    return(ax, ax2)
#    
#def plot_density(filename, binc, hist_dicts, dens=True):
#    _set_plotting_env(width=3.37,height=3.37/ golden_ratio *1.5/2,\
#                   lrbt=[0.18,0.88,0.25,0.95],fsize=9.0)
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax, ax2 = _plot_density(binc, hist_dicts, ax, dens)
#    if ax2 != None:
#        ax2.set_ylabel(r'density water (g/cm$^3$)')
#    #
#    ax.set_xlabel(r'z ($\mathrm{\AA}$)')
#    ax.set_ylabel(r'density (mol/cm$^3$)')
#    ax.legend(loc='best',prop={'size':7})
#    ax.set_xlim(0.0,ax.get_xlim()[1])
#    writefig(filename)
#   #matplotlibhelpers.write(filename,folder='output',\
#   #    write_info = False,write_png=False,write_pdf=True,write_eps=False)

#############################################################
#############################################################
#############################################################


def _sort_sub_runs(folders):
    bkeys = list(set(['_'.join(f.split('_')[:-1]) for f in folders]))
    fdict = {bk:[k for k in folders if '_'.join(k.split('_')[:-1]) == bk] \
                                                        for bk in bkeys}
    return(fdict)

def _adjust_density_to_metal(binc, hist_dicts, metal):
    # find where metal is in density profile
    indm = np.where(hist_dicts[metal] != 0.0)[0]
    binc = binc[indm[-1]+1:]
    for k in hist_dicts:
        hist_dicts[k] = hist_dicts[k][indm[-1]+1:]
    return(binc, hist_dicts)

def _av_dens(folders):
    sdens = {}
    for sub in folders:
        denspkl = '%s/density_dat.pkl'%sub
        dens = _load_pickle_file(denspkl)
        dens['binc'], dens['hist_dicts'] = _adjust_density_to_metal(\
            dens['binc'], dens['hist_dicts'], sub[:2])
        # sum-up values
        hist_dicts = dens['hist_dicts']; hist_dicts.update({'binc':dens['binc']})
        for k in hist_dicts.keys():
            if k not in sdens:
                sdens.update({k:np.zeros(hist_dicts[k].shape)})
               # first profile determines shape (little data lost - if at all)
            lk = min(sdens[k].size, hist_dicts[k].size)
            sdens[k][:lk] += hist_dicts[k][:lk]
    for k in sdens:
        sdens[k] /= len(folders)
    return(sdens)

def _analyze_density(indens):
    sdens = deepcopy(indens)
    # delta ind for 5 AA
    id5 = np.where(sdens['binc']-sdens['binc'][0] > 2.0)[0][0]
    # primitive identification of 1st peak density
    dat = {}
    dh2o = {'Ow':1/18., 'Hw':2/18.}
    for wk in ['Ow', 'Hw']:
        ipk = sdens[wk].argmax()
        ilay = ipk+sdens[wk][ipk:ipk+id5].argmin()
        # integrate # of H2O
        igr = np.trapz(sdens[wk][:ilay], sdens['binc'][:ilay])
        dat.update({wk:[ilay, ipk, igr/dh2o[wk]]})
        # peak onset
        #print(wk, sdens['binc'][np.where(sdens[wk] != 0.0)[0][0]] - sdens['binc'][0])
    return(dat)

def _isolate_pk(sdens,ipk,k):
    x = sdens['binc'][ipk]-sdens['binc'][0]
    y = sdens[k][ipk]
    return(np.array([x,y]))

def plot_density_fancy(filename, folders):
    colors = {'Ow':'r','Hw':'c'}; tshift = {'Ow':0.7, 'Hw':-1.7}; dy = -0.03
    surftag = {'Cu111': 'Cu(111)', 'Cu211':'Cu(211)', 'Au111':'Au(111)', \
        'Pt111': 'Pt(111)', 'Cu211-6x4':'Cu(211)', 'Cu211-3x4':'Cu(211)', 'Pt111-6x4':'Pt(111)'}
    dh2o = {'Ow':1/18., 'Hw':2/18.}
    
    ref = _sort_sub_runs(folders)
    r = list(ref.keys())[0]

    # make density
    sdens = _av_dens(folders)
    
    # for normalization
    aa = read(r+'_02/POSCAR')
    Acell = aa.get_volume()/aa.get_cell()[2,2]
            
    _set_plotting_env(width=3.37,height=3.37/ golden_ratio,\
                   lrbt=[0.18,0.95,0.20,0.95],fsize=9.0)
    fig, ax = plt.subplots(1,1)
    axes = _plot_density(sdens['binc'], {k:sdens[k] for k in sdens if k != 'binc'}, ax=ax)
    axes[0].annotate(r'%s'%surftag[r.split('_')[0]], xy=(0.45,0.7), xycoords='axes fraction',size=9)
            
    dpk = _analyze_density(sdens)
    print(50*'#')
    print(r, [[k, sdens['binc'][dpk[k][1]]-sdens['binc'][0], dpk[k][2]] for k in dpk])

    # determine average density of clean sample
    x0 = sdens['binc'][0]
    iwl = np.where(sdens['binc']-x0 > 9)[0][0]
    ist = np.where(sdens['Ow'] > 1e-2)[0][0] #onset of noticable density
    dens = np.trapz(sdens['Ow'][:iwl], sdens['binc'][:iwl]) / (9.0-(sdens['binc'][ist]-x0)) 
    dens /= dh2o['Ow']
    print('average density %s excluding vacuum interface: %.2f'%(r, dens))
    
    lim_int = [dpk[k][0] for k in dpk]
    lim_int = int(np.mean(lim_int))
    
    nfac = {'Ow':1.0, 'Hw':2.0}
    for k in dpk: #plot first layer
        axes[0].fill_between(sdens['binc'][:lim_int]-sdens['binc'][0], 0, sdens[k][:lim_int], color=colors[k], alpha=0.3)
        pxy = _isolate_pk(sdens,dpk[k][1],k)
        ist = np.where(sdens[k] > 1e-2)[0][0] #onset of noticable density
        Nh2o = dh2o[k]*(sdens['binc'][lim_int]-sdens['binc'][ist])*1e-8 #normalization of water density
        Nnum = 1e-9 # unit 1e-9 mol/cm2
        Ncell = Acell * (1e-8**2.0) # unit-cell area in cm2
        Navg = 6.02214e23
        igr = np.trapz(sdens[k][:lim_int], sdens['binc'][:lim_int]*1e-8) * Ncell *Navg / nfac[k] # / Nnum #Nh2o
        axes[0].annotate(r'%.1f'%igr, xy=pxy+np.array([tshift[k]/10.0,dy]), xytext=pxy+np.array([tshift[k], -0.01]), size=8, color=colors[k],
            arrowprops=dict(arrowstyle="-",connectionstyle="arc",color=colors[k]))
    # make pretty
    axes[0].set_xlabel(r'z ($\mathrm{\AA}$)')
    axes[0].set_ylabel(r'density (mol/cm$^3$)')
    leg = axes[0].legend(loc=1,prop={'size':7},framealpha=1.0,facecolor='w')
    [axes[i].set_zorder(99) for i in range(1)]#2)]
    writefig(filename)
 
def _process_densities(filename, folders, write_eps=False):
    surftag = {'Cu111': 'Cu(111)', 'Cu211':'Cu(211)', 'Au111':'Au(111)', \
        'Pt111': 'Pt(111)', 'Cu211-6x4':'Cu(211)', 'Cu211-3x4':'Cu(211)', 'Pt111-6x4':'Pt(111)'}
    colors = {'Ow':'r','Hw':'c'}; tshift = {'Ow':0.7, 'Hw':-1.7}; dy = -0.03; #tshift = {'Ow':0.7, 'Hw':-1.5}; dy = -0.05
    # presort keys
    aorder = ['CO','CHO','COH','OCCHO','OH','OOH']
    ads = [k for k in folders if np.any([k.find(m) != -1 for m in aorder])]
    ads = _sort_sub_runs(ads)
    ref = [k for k in folders if np.all([k.find(m) == -1 for m in aorder])]
    ref = _sort_sub_runs(ref)

    # fine-tuning
    i_shift = {'Cu211_15H2O':2}; txshift = {'Pt111_24H2O':np.array([0.15,-0.03]), 'Cu211_15H2O':np.array([0.2,0.0])}
    
    dh2o = {'Ow':1/18., 'Hw':2/18.}
    
    # all densities sampled from 5e3 onward
    for r in ref:
        aa = read(r+'_02/POSCAR')
        Acell = aa.get_volume()/aa.get_cell()[2,2]
        rads = [a for a in ads if '_'.join(a.split('_')[:-1]) == r]
        rads = [r+'_'+a for a in aorder if r+'_'+a in rads]
        if len(rads) > 0:
            # avering densities
            sdens = _av_dens(ref[r])
            
            _set_plotting_env(width=3.37,height=3.37/ golden_ratio *1.5,\
                           lrbt=[0.18,0.95,0.15,0.95],fsize=9.0)
            if r == 'Cu211_15H2O':
                _set_plotting_env(width=3.37,height=3.37/ golden_ratio *1.8,\
                               lrbt=[0.18,0.95,0.12,0.98],fsize=9.0)

            #fig, axes = plt.subplots(len(ref[r])+1,1)#,figsize=np.array([6.4,4.8])*nplot/6.0)
            fig, axes = plt.subplots(len(rads)+1,1)#,figsize=np.array([6.4,4.8])*nplot/6.0)
            axes[0] = _plot_density(sdens['binc'], {k:sdens[k] for k in sdens if k != 'binc'}, ax=axes[0])
            axes[0][0].annotate(r'%s'%surftag[r.split('_')[0]], xy=(0.45,0.7), xycoords='axes fraction',size=9)
            
            dpk = _analyze_density(sdens)
            print(50*'#')
            print(r, [[k, sdens['binc'][dpk[k][1]]-sdens['binc'][0], dpk[k][2]] for k in dpk])

            # determine average density of clean sample
            x0 = sdens['binc'][0]
            iwl = np.where(sdens['binc']-x0 > 9)[0][0]
            ist = np.where(sdens['Ow'] > 1e-2)[0][0] #onset of noticable density
            dens = np.trapz(sdens['Ow'][:iwl], sdens['binc'][:iwl]) / (9.0-(sdens['binc'][ist]-x0)) 
            dens /= dh2o['Ow']
            print('average density %s excluding vacuum interface: %.2f'%(r, dens))

            dpk_ads = {ra:_analyze_density(_av_dens(ads[ra])) for ra in rads}
            lim_int = [dpk[k][0] for k in dpk] + [dpk_ads[ra][k][0] for ra in dpk_ads for k in dpk_ads[ra]]
            lim_int = int(np.mean(lim_int))
            if r in i_shift:
                lim_int += i_shift[r]
            
            txs = np.zeros(2)
            if r in txshift:
                txs = txshift[r]
            
            nfac = {'Ow':1.0, 'Hw':2.0}
            for k in dpk: #plot first layer
                axes[0][0].fill_between(sdens['binc'][:lim_int]-sdens['binc'][0], 0, sdens[k][:lim_int], color=colors[k], alpha=0.3)
                #axes[0][0].fill_between(sdens['binc'][:dpk[k][0]]-sdens['binc'][0], 0, sdens[k][:dpk[k][0]], color=colors[k], alpha=0.3)
                pxy = _isolate_pk(sdens,dpk[k][1],k)
                ist = np.where(sdens[k] > 1e-2)[0][0] #onset of noticable density
                Nh2o = dh2o[k]*(sdens['binc'][lim_int]-sdens['binc'][ist])*1e-8 #normalization of water density
                Nnum = 1e-9 # unit 1e-9 mol/cm2
                Ncell = Acell * (1e-8**2.0) # unit-cell area in cm2
                Navg = 6.02214e23
                #igr = np.trapz(sdens[k][:lim_int], sdens['binc'][:lim_int]) / dh2o[k]
                igr = np.trapz(sdens[k][:lim_int], sdens['binc'][:lim_int]*1e-8) * Ncell *Navg / nfac[k] # / Nnum #Nh2o
                #axes[0][0].annotate(r'%.1f'%dpk[k][2], xy=pxy+np.array([tshift[k]/4.0,dy]), xytext=pxy+np.array([tshift[k], -0.01]), size=8, color=colors[k],
                axes[0][0].annotate(r'%.1f'%igr, xy=pxy+np.array([tshift[k]/10.0,dy]), xytext=pxy+np.array([tshift[k], -0.01])+txs, size=8, color=colors[k],
                    arrowprops=dict(arrowstyle="-",connectionstyle="arc",color=colors[k]))

            n = 1
            for ra in rads:
                radens = _av_dens(ads[ra])
                axes[n] = _plot_density(radens['binc'], {k:radens[k] for k in radens if k != 'binc'}, ax=axes[n])
                axes[n][0].annotate(r'%s'%ra.split('_')[-1], xy=(0.45,0.7), xycoords='axes fraction',size=9)
            
                dpk = _analyze_density(radens)
                print(ra, [[k, sdens['binc'][dpk[k][1]]-sdens['binc'][0], dpk[k][2]] for k in dpk])
                for k in dpk: #plot first layer
                    axes[n][0].fill_between(radens['binc'][:lim_int]-radens['binc'][0], 0, radens[k][:lim_int], color=colors[k], alpha=0.3)
                    #axes[n][0].fill_between(radens['binc'][:dpk[k][0]]-radens['binc'][0], 0, radens[k][:dpk[k][0]], color=colors[k], alpha=0.3)
                    pxy = _isolate_pk(sdens,dpk[k][1],k)
                    ist = np.where(radens[k] > 1e-2)[0][0] #onset of noticable density
                    Nh2o = dh2o[k]*(radens['binc'][lim_int]-radens['binc'][ist])*1e-8 #normalization of water density
                    Nnum = 1e-9 # unit 1e-9 mol/cm2
                    if ra.find('COH') != -1 and k == 'Hw':
                        pxy = np.array([2.0,0.15])
                    #igr = np.trapz(radens[k][:lim_int], radens['binc'][:lim_int]) / dh2o[k]
                    #igr = np.trapz(radens[k][:lim_int], radens['binc'][:lim_int]*1e-8) / Nnum #Nh2o
                    igr = np.trapz(radens[k][:lim_int], radens['binc'][:lim_int]*1e-8) * Ncell *Navg / nfac[k] # / Nnum #Nh2o
                    #axes[n][0].annotate(r'%.1f'%dpk[k][2], xy=pxy+np.array([tshift[k]/4.0,dy]), xytext=pxy+np.array([tshift[k], -0.01]), size=8, color=colors[k],
                    axes[n][0].annotate(r'%.1f'%igr, xy=pxy+np.array([tshift[k]/10.0,dy]), xytext=pxy+np.array([tshift[k], -0.01])+txs, size=8, color=colors[k],
                        arrowprops=dict(arrowstyle="-",connectionstyle="arc",color=colors[k]))
                
                n += 1
            print(50*'#')
           #n = 1
           #for sub in ref[r]:
           #    # load and adjust size to metal
           #    denspkl = '%s/density_dat.pkl'%sub
           #    dens = _load_pickle_file(denspkl)
           #    dens['binc'], dens['hist_dicts'] = _adjust_density_to_metal(\
           #        dens['binc'], dens['hist_dicts'], sub[:2])
           #    axes[n] = _plot_density(dens['binc'], dens['hist_dicts'], ax=axes[n])
           #    n += 1
            
            # make pretty
            #for s in range(2):
            for s in range(1):
                #lims = np.array([list(ax[s].get_ylim()) for ax in axes])
                #mlims = (lims[:,0].min(),lims[:,1].max())
                lims = np.array([list(ax[s].get_xlim()) for ax in axes])
                #mlims = (lims[:,0].min(),lims[:,1].min())
                mlims = (0.0,lims[:,1].min())
                [ax[s].set_xlim(mlims) for ax in axes]
                [ax[s].set_ylim(0.0,0.27) for ax in axes]
            if r == 'Cu211_15H2O':
                [ax[0].yaxis.set_major_locator(MultipleLocator(0.1)) for ax in axes]
            [axes[i][0].set_xticklabels([]) for i in range(len(axes)-1)]
            axes[-1][0].set_xlabel(r'z ($\mathrm{\AA}$)')
            axes[len(axes)//2][0].set_ylabel(r'density (mol/cm$^3$)')
            if len(axes)%2 == 0:
                axes[len(axes)//2][0].yaxis.set_label_coords(-0.12, 1.0)
            #axes[len(axes)//2][0].yaxis.set_label_coords(-0.15, 1.0)
            #axes[len(axes)//2][1].set_ylabel(r'density water (g/cm$^3$)')
            leg = axes[0][0].legend(loc=1,prop={'size':7},framealpha=1.0,facecolor='w')
            [axes[0][i].set_zorder(99) for i in range(1)]#2)]
            plt.subplots_adjust(hspace=0.0)
            writefig(filename+'_'+r, write_eps)
    
def plot_msd_individual(filename, msd_dat, write_eps=False):
    _set_plotting_env(width=3.37,height=3.37/ golden_ratio, # *1.5,\
                   lrbt=[0.17,0.95,0.17,0.95],fsize=9.0)
    subtag = {'av0.5':'av05', 'msd':'raw'}
    c = {'ads':tumcs['darkgray'], 'water':tumcs['tumblue']}
    msdk = ['x','y','z','xy','xz','yz','xyz']
    for sub in ['av0.5', 'msd']:
        fig, axes = plt.subplots(2,1)
        for f in msd_dat:
            da = msd_dat[f]['ads'][sub]; dw = msd_dat[f]['water'][sub]
            x = np.arange(da[:,0].size)*0.1 # time in ps
            
            axes[0].plot(x, da[:,msdk.index('xy')], color=c['ads'], ls='-')
            axes[0].plot(x, da[:,msdk.index('xyz')], color=c['ads'], ls='--')
            
            axes[1].plot(x, dw[:,msdk.index('xy')], color=c['water'], ls='-')
            axes[1].plot(x, dw[:,msdk.index('xyz')], color=c['water'], ls='--')
 
        # make pretty
        axes[0].set_xticklabels([])
        plt.subplots_adjust(hspace=0.0)
        axes[1].set_xlabel(r'time (ps)')
        axes[0].set_ylabel(r'msd ($\mathrm{\AA}^2$)')
        axes[0].yaxis.set_label_coords(-0.15, 0.0)
        axes[0].plot(np.nan, np.nan, color='k', ls='-', label=r'xy')
        axes[0].plot(np.nan, np.nan, color='k', ls='--', label=r'xyz')
        leg = axes[0].legend(loc=2,prop={'size':7},framealpha=1.0,facecolor='w')
            
        writefig(filename+'_'+subtag[sub], write_eps)
    
def plot_diffusion_coefficients(filename, rdat, write_eps=False):
    _set_plotting_env(width=3.37,height=3.37/ golden_ratio,# *1.5,\
                   lrbt=[0.14,0.95,0.22,0.95],fsize=9.0)
    c = {'ads':tumcs['darkgray'], 'water':tumcs['tumblue']}
    surftag = {s:'%s(%s)'%(s[:2],s[2:]) for s in ['Cu211','Cu111','Au111','Pt111']}
    
    order = ["Cu211_15H2O%s"%a for a in ['','_CO','_CHO','_COH','_OCCHO','_OH']] + ['']
    order += ['Cu111_24H2O%s'%a for a in ['','_CO','_OH']] + ['']
    for s in ['Au111','Pt111']:
        order += ['%s_24H2O%s'%(s,a) for a in ['','_CO','_OH','_OOH']] + ['']
    #order += ['%s_24H2O%s'%(s,a) for s in ['Cu111','Au111','Pt111'] for a in ['','_OH','_CO']]
    water = {'_'.join(f.split('_')[:2]):[] for f in rdat}
    dat = {f:rdat[f] for f in rdat if f in order}
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for f in dat:
        for sub in ['ads','water']:
            # choose x
            x = order.index(f)
            if sub == 'water' and len(f.split('_')) > 2:
                x = order.index('_'.join(f.split('_')[:-1]))
                water['_'.join(f.split('_')[:2])] += dat[f][sub]
           #if f.split('_')[0] == 'Cu111' and f.split('_')[2] == 'OH' and sub == 'ads':
           #    print(dat[f][sub])
            elif np.mean(dat[f][sub]) != 0.0 and sub == 'ads':
                ax.errorbar(x, np.mean(dat[f][sub]), yerr=np.std(dat[f][sub])/2., \
                    color=c[sub], marker='o')
    for f in water:
        x = order.index(f)
        ax.errorbar(x, np.mean(water[f]), yerr=np.std(water[f]), \
            color=c['water'], marker='o')
        ax.plot([x-0.7, x+0.7], [0.23, 0.23], ls=':', lw=2, color=c['water'], alpha=0.7)
    
    # plot partitioning: vlines + annotations
    csurf = {'Cu211':tumcs['tumred'], 'Cu111':tumcs['tumorange'],\
        'Au111':tumcs['acc_yellow'], 'Pt111':tumcs['darkgray']}
    ylims = ax.get_ylim(); xl = [l for l in range(len(order)) if order[l] == '']
    [ax.axvline(i, color='k', lw=1) for i in xl[:-1]]
    bbox_props = dict(boxstyle="round,pad=0.15", fc="w", ec=tumcs['darkgray'], lw=2)
    ys = [0.6,0.6,0.6,0.6]
    ys = [0.8,0.8,0.8,0.8]
    #ys = [1.55,1.55,1.55,1.55]
    [ax.text(xl[l]-3.9, ys[l] ,r'%s'%(surftag[order[xl[l]-1].split('_')[0]]), \
        color=csurf[order[xl[l]-1].split('_')[0]]) for l in range(len(xl))]
        #color=csurf[order[xl[l]-1].split('_')[0]], bbox=bbox_props) for l in range(len(xl))]

    order[3] = 'Cu211_15H2O_  COH'
    order[5] = 'Cu211_15H2O_      OH   '
    plt.xticks(np.arange(len(order)),2*[''] + [''.join(o.split('_')[2:]) for o in order], rotation=50.)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_ylabel(r'$D^*$ ($\mathrm{\AA^2}$/ps)')
    ax.set_ylim(-0.05,0.9)
    ax.set_xlim(-1.0, len(order)-1)
    for sub in ['ads','water']:
        ax.plot(np.nan, np.nan, color=c[sub], marker='o', label=r'%s'%sub)
    leg = ax.legend(loc=7,prop={'size':7},framealpha=1.0,facecolor='w')
        
    writefig(filename, write_eps)

def _plot_h2o_adsorption(filename, dat, write_eps=False):
    # presort to data i want
    dat = {k:dat[k] for k in dat if np.all([k.find(tg) == -1 for tg in ['K+','Li+','Na+','3x4']])}
    surfs = ['Cu211','Cu111','Au111','Pt111']; surftag = {s:'%s(%s)'%(s[:2],s[2:]) for s in surfs}
    ads = ['','CO','CHO','COH','OCCHO','OH','OOH']
    nh2o = {'211':'15H2O', '111':'24H2O'}
    
    # for plotting
    _set_plotting_env(width=3.37,height=3.37/ golden_ratio,# *1.5,\
                   lrbt=[0.16,0.98,0.22,0.95],fsize=9.0)
    csurf = {'Cu211':tumcs['tumred'], 'Cu111':tumcs['tumorange'],\
        'Au111':tumcs['acc_yellow'], 'Pt111':tumcs['darkgray']}
    fig = plt.figure(); x = 0; lvs = []; atgs = []; stgs = []
    ax = fig.add_subplot(111)
    for s in surfs:
        for a in ads:
            k = '_'.join([s,nh2o[s[2:]],a])
            if a == '':
                k = k[:-1]; stgs.append(x)
            if k in dat:
                ax.bar(x, dat[k][0], yerr=dat[k][1]/2., \
                    color=csurf[s], capsize=2, width=0.7)
                x += 1
                atgs.append(a)
        lvs.append(x); atgs.append('')
        x +=1
    [ax.axvline(lv, ls='-', color='k', lw=1.0) for lv in lvs[:-1]]
    
    bbox_props = dict(boxstyle="round,pad=0.15", fc="w", ec=tumcs['darkgray'], lw=2)
    ys = [1.55,1.55,1.55,1.55]; xx = [0.0,0.2,0.0,0.0]
    [ax.text(stgs[i]-0.7-xx[i], 2.2 ,r'%s'%(surftag[surfs[i]]), color=csurf[surfs[i]],) #bbox=bbox_props) 
        for i in range(len(surfs))]
    atgs[5] = '    OH     '; atgs[3] = '  COH'
    plt.xticks(np.arange(len(atgs)),atgs, rotation=50.)
    ax.set_ylabel(r'$\langle n\mathrm{_{ad}^{H_2O}} \rangle$')
    #ax.set_ylim(-0.1,1.8)
    writefig(filename, write_eps)

def _get_ledge_metal_slab(f, metal):
    denspkl = '%s/density_dat.pkl'%f
    dens = _load_pickle_file(denspkl)
    indm = np.where(dens['hist_dicts'][metal] != 0.0)[0]
    binc = dens['binc'][indm[-1]]
    return(binc)

def _plot_orientation_distribution(filename, dat, z0=0):
    # sum together every d=20 --> roughly 5 AA
    zorder = np.sort(list(dat['h2o'].keys())); d = 20
    rzorder = {np.mean(zorder[i:min(i+d, zorder.size)]):zorder[i:min(i+d, zorder.size)] \
        for i in range(0,len(zorder),d)}
    abins = np.arange(0,180+1,2); acntr = (abins[:-1]+abins[1:]) /2.0; dz=abins[1]-abins[0]
    
    _set_plotting_env(width=3.37,height=3.37/ golden_ratio *1.5,\
                   lrbt=[0.14,0.94,0.15,0.95],fsize=9.0)
    lss = {'h2o':':', 'OH':'-'}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for sub in dat:
        rdat = {zz:[] for zz in rzorder}
        for zz in rzorder:
            for z in rzorder[zz]:
                rdat[zz] += dat[sub][z]
        # throw out len == 0 zero ones 
        rdat = {k:rdat[k] for k in rdat if len(rdat[k]) != 0}

        # make hist of data + plot it
        for z in rdat:
            d, bins = np.histogram(rdat[z], abins, density=True)
            ax.plot(acntr, (d/d.max())*2.0+z-z0, ls=lss[sub])

    ax.plot(np.nan, np.nan, ls=lss['h2o'], label=r'$m_{H_2O}$')
    ax.plot(np.nan, np.nan, ls=lss['OH'], label=r'$b_{OH}$')
    leg = ax.legend(loc=1,prop={'size':7},framealpha=1.0,facecolor='w')

    ax2 = ax.twinx()
    ax2.set_ylabel(r'$I_{\alpha / \beta}$ (a.u.)')
    ax2.set_yticks([])
        
    ax.set_ylabel(r'z ($\mathrm{\AA}$)')
    ax.set_xlabel(r'angle vs. surface normal ($^\circ$)')
    writefig('water_orientation/'+filename)

#def writefig(filename, write_eps=False):
#    folder = 'output'
#    if not os.path.isdir(folder):
#        os.makedirs(folder)
#    fileloc = os.path.join(folder, filename)
#    print("writing {}".format(fileloc+'.pdf'))
#    plt.savefig(fileloc+'.pdf')
#    if write_eps:
#        print("writing {}".format(fileloc+'.eps'))
#        plt.savefig(fileloc+'.eps')

def get_run_aimd_wf(wpath):
    wf_data = read_sp_wf_data(wpath)
    approx_wf = None
    if len(wf_data) > 0:
        dplc_dat = get_vasp_iterative(wpath, _read_xml_dipole_corrections)['dipole_correction']
        # fitting may come with a large error!
        fvals = np.array([[dplc_dat[int(k)],wf_data[k]] for k in wf_data])
        popt, func = get_linear_fit(fvals[:,0],fvals[:,1])
        approx_wf = func(dplc_dat,*popt)
  ###print([func(dplc_dat[int(k)],*popt)-wf_data[k] for k in wf_data])
  ##print('deviation',np.std([func(dplc_dat[int(k)],*popt)-wf_data[k] for k in wf_data]))
  ##wfas = []
  ##for s in wf_data:
  ##    wf = wf_data[s]; dp0 = dplc_dat[int(s)]
  ##    wfa = (dplc_dat - dp0)*2 + wf
  ##    wfas.append(wfa)
  ##mwfa = np.mean(np.array(wfas),axis=0)
  ##wdiffs = [wf_data[k] - mwfa[int(k)] for k in wf_data]
    return(wf_data, approx_wf)

def get_run_dvac_sp(wpath, dat, rH2O):
    # NOTE: internal H2O structure averaged out or need to be included?
    #       would need other singlepoint of only water (plot and see)
    #       its crucial to note that the STD of sp_vacuum is only 0.097 eV
    #       while the difference with water is 1.0185 eV (likely crazy noise)
    #       however most of the noise comes from the first step!!! - move this remark somewhere else
    
    # post-process single-point vacuum; e-diff to solvated
    sp_vac = read_sp_nosolv_data(wpath)
    
    # also include vac H2O reference
    #nH2O = len(identify_water(read(wpath+"/atoms.traj",-1))) # NOT WORKING
 ###nH2O = int(wpath.split('/')[-1].split('_')[1][:2])
 ###eH2O = rH2O*nH2O
 ###
 ###dsp_vac = np.array([[t, dat['epot'][int(t)]-sp_vac[t]-eH2O] \
 ###                        for t in np.sort(list(sp_vac.keys()))])
    # NOTE: this is only the single points - approx. to vacuum traj
    dsp_vac = np.array([[t, sp_vac[t]] for t in np.sort(list(sp_vac.keys()))])
    return(dsp_vac)

def read_sp_nosolv_data(wpath):
    sppath = wpath+"/singlepoints_nosolvent"; pklfile = "nosol_sp.pkl"
    dat = {}; folders = []; warn = False
    if os.path.isdir(sppath):
        folders = [sppath+'/'+f for f in os.listdir(sppath) if f[:3] == 'sp_']
    if os.path.isfile(sppath+"/%s"%pklfile):
        dat = _load_pickle_file(sppath+"/%s"%pklfile)

    if len(folders) > 0 and len(folders) > len(dat.keys()):
        print('post-processing nosolvent data in %s'%(wpath))
        for f in folders: # if no entry of folder in dat but output there
            if os.path.isfile(f+'/vasprun.xml') and check_vasp_opt_converged(f):
                dat.update({float(f[-5:]):_read_xml_singlepoint(f+"/vasprun.xml")['epot']})
            else:
                warn = True
        _write_pickle_file(sppath+"/%s"%pklfile, dat)
    if warn:
        print(80*'#'+'\n'+'Warning %s nosolvent is incomplete'%wpath+'\n'+80*'#')
    return(dat)

def get_linear_fit(x,y):
    f_lin = lambda x, a, b: a + b*x
    popt, pcov = curve_fit(f_lin, x[:], y[:], p0=[y[0],y[-1]-y[0]], maxfev=100000)
    return(popt, f_lin)

def _read_xml_dipole_corrections(filename):
    tstr = popen("grep '%s' -A 10 %s"%("nosekinetic",filename)).read().split()
    dplcrct_iter1 = np.array([float(s) for s in tstr[29::32]])
    
    tstr = popen("grep '%s' %s"%("dipole",filename)).read().split()
    dplcrct_all = np.array([float(s) for s in tstr[5::8]])
    dplcrct = []
    for d1 in dplcrct_iter1:
        ind = np.where(dplcrct_all == d1)[0]-1
        dplcrct.append(dplcrct_all[ind][0])
    return({'dipole_correction':np.array(dplcrct)})

def get_vasp_iterative(wpath, func):
    data = []
    while os.path.isfile(wpath+"/OUTCAR") or os.path.isfile(wpath+"/vasprun.xml"):
        if os.path.isfile(wpath+"/vasprun.xml"):
            data.append(func(wpath+"/vasprun.xml"))
        wpath += '/restart'
    dat = _join_data_list(data)
    return(dat)
        
def _plot_potential(z, d0, marks=[]):
    _set_plotting_env(width=3.37,height=3.37/ golden_ratio *1.5,\
                   lrbt=[0.16,0.94,0.15,0.95],fsize=9.0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(z,d0,ls='-')
    for mk in marks:
        ax.axvline(mk, ls=':', color='orange', lw=1.0)

    ax.set_ylabel(r'$\Phi$ (eV)')
    ax.set_xlabel(r'z ($\mathrm{\AA}$)')
    writefig('potential')

#############################################################
################## NOTE: already worked in ##################
##############################################################
    
#def read_sp_wf_data(wpath):
#    #a = read(wpath+'/atoms.traj') #one snapshot for volume
#    sppath = wpath+"/singlepoints_wf"
#    dat = {}; folders = []; warn = False
#    if os.path.isdir(sppath):
#        folders = [sppath+'/'+f for f in os.listdir(sppath) if f[:3] == 'sp_']
#    if os.path.isfile(sppath+"/wfs.pkl"):
#        dat = _load_pickle_file(sppath+"/wfs.pkl")
#
#    if len(folders) > 0 and len(folders) > len(dat.keys()):
#        print('post-processing wf data in %s'%(wpath))
#        for f in folders: # if no entry of folder in dat but output there
#            if float(f[-5:]) not in dat and \
#                (os.path.isfile(f+'/LOCPOT') or os.path.isfile(f+'/LOCPOT.gz')):
#                wf = get_vasp_wf(f)
#                dat.update({float(f[-5:]):wf})
#            else:
#                warn = True
#        _write_pickle_file(sppath+"/wfs.pkl", dat)
#    if warn:
#        print(80*'#'+'\n'+'Warning %s wf is incomplete'%wpath+'\n'+80*'#')
#    return(dat)
#
#def _unpack_files(wdir, files):
#    for f in files:
#        fn_packed = wdir+'/%s.gz'%f; fn_unpacked = wdir+'/%s'%f
#        if os.path.isfile(fn_packed):
#            with gzip.open(fn_packed, 'rb') as f_in, open(fn_unpacked, 'wb') as f_out:
#                f_out.writelines(f_in)
#            os.remove(fn_packed)
#
#def get_vasp_wf(f, **kwargs):
#    _unpack_files(f,['OUTCAR','LOCPOT']) #in case its packed
#
#    d0, a = _read_potential_vasp(f)
#    ind_vac = _determine_vacuum_reference(d0, a, **kwargs)[0]
#    
#    # fermi energy
#    ef = float(get_efermi(f+"/OUTCAR"))
#    
#    wf = d0[ind_vac] - ef
#    return(wf)
#
#def _read_potential_vasp(f):
#    a = read(f+'/OUTCAR')
#    # local potential
#    c=vasp.VaspChargeDensity(filename=f+'/LOCPOT')
#    c.read(filename=f+'/LOCPOT')
#    dens=c.chg[0]*a.get_volume() # ASE factors wronlgy via volume
#    d0=np.mean(np.mean(dens,axis=0),axis=0)
#    return(d0, a)
#
#def _determine_vacuum_reference(d0, a, viz=False, gtol=1e-3):
#    # gtol == tolerance for when potential is 'converged'/flat in V/AA
#    # vacuum region - first guess
#    z=np.linspace(0,a.cell[2,2],len(d0))
#    z_vac = find_max_empty_space(a,edir=3)
#    ind_vac = np.absolute(z - z_vac).argmin()
#
#    # double potential for simpler analysis at PBC
#    d02 = np.hstack((d0,d0))
#    # investigate at gradient
#    gd02 = np.absolute(np.gradient(d02))
#    iv2 = ind_vac+d0.size
#    # search max gradient as center of dipole correction withint 3 AA
#    diA = np.where(z > 1.0)[0][0]
#    imax = iv2-diA*2 + gd02[iv2-diA*2:iv2+diA*2].argmax()
#
#    # walk from imax to find minimum
#    # ibfe/iafr --> before/after dipole correction along z-axis
#    ibfe = np.where(gd02[imax-diA*3:imax] < gtol)[0] + imax-diA*3
#    iafr = np.where(gd02[imax:imax+diA*3] < gtol)[0] + imax
#
#    if ibfe.size == 0 or iafr.size == 0:
#        print('###\nproblematic WF gradient\n###')
#        ibfe = iafr = [ind_vac]
#        # TODO: fix problems with 211 --> seemingly max gradient cannot be reliable found
#        #raise Exception('problematic WF gradient')
#    ibfe = ibfe[-1]%d0.size; iafr = iafr[0]%d0.size
#    
#    if viz:
#        ###_plot_potential(z, d0, marks=[z[ind_vac], z[imax%d0.size]])
#        _plot_potential(z, d0, marks=[z[imax%d0.size], z[ibfe], z[iafr]])
#    return([ibfe, iafr])
    
#def get_efermi(ofile):
#    with open(ofile,'r') as f:
#        lines = f.readlines()
#    ef = lines_key(lines,'E-fermi',0,2,rev=True,loobreak=True)
#    return(ef)

#def check_vasp_opt_converged(cdir):
#    _unpack_files(cdir,['OUTCAR','OSZICAR']) #in case its packed
#    with open(cdir+'/OUTCAR','r') as f:
#        olines = f.readlines()
#    with open(cdir+'/OSZICAR','r') as f:
#        zlines = f.readlines()
#    
#    # finished OUTCAR-file ?
#    check = True and lines_key(olines,\
#        'General timing and accounting informations',0,1,rev=True,loobreak=True) == 'timing'
#    
#    # last iteration didn't hit NELM ?
#    nscflast = int(lines_key(zlines,'DAV:',0,1,rev=True,loobreak=True))
#    nelm = 0
#    if lines_key(olines,'NELM   =',0,2,rev=False,loobreak=True) != False:
#        nelm = int(lines_key(olines,'NELM   =',0,2,rev=False,loobreak=True)[:-1])
#    check = check and nscflast < nelm
#    
#    # if geo-opt converged?
#    #fconv = float(lines_key(olines,'EDIFFG =',0,2,rev=False,loobreak=True))
#    fs = [float(line.split()[2]) for line in zlines if line.find('F=') != -1]
#    #alist = read(cdir+'/OUTCAR',':'); fs = [a.get_forces().max() for a in alist]
#    #check = check and (len(fs) > 1 and abs(fconv) > abs(fs[-1]-fs[-2]) or len(fs) == 1)
#    check = check and (lines_key(olines,\
#        "reached required accuracy - stopping structural energy minimisation",\
#        0,1,rev=True,loobreak=True) == 'required' or len(fs) == 1)
#    return(check)

#############################################################
#############################################################
##############################################################
        
def plot_wf_vs_time(filename,wfdat,wfapprox=np.zeros([])):
    _set_plotting_env(width=3.37,height=3.37/ golden_ratio,\
                   lrbt=[0.18,0.88,0.25,0.95],fsize=9.0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if wfapprox.size > 0:
        t = np.arange(wfapprox.size)/1000.
        ax.plot(t[::200],wfapprox[::200],ls='--',lw=0.5,color='k',label=r'approx. WF')
    
    wfxy = np.array([[s/1000.,wfdat[s]] for s in wfdat])
    ax.plot(wfxy[:,0],wfxy[:,1],ls='None',marker='d',color='r',label=r'WF')

    ax.set_xlabel(r'simulation time (ps)')
    ax.set_ylabel(r'work function / eV')
    ax.legend(loc=2,prop={'size':7})
    writefig(filename)

def plot_esol_vs_time(filename,dsol):
    #presort data to keys
    fkeys = list(set(['_'.join(k.split('_')[:2]) for k in dsol.keys()]))
    fkeys = {f:[sf for sf in dsol.keys() \
        if '_'.join(sf.split('_')[:2]) == f and len(dsol[sf]) > 0] for f in fkeys}
    fkeys = {f:fkeys[f] for f in fkeys if len(fkeys[f]) > 0}
    
    ## plotting
    _set_plotting_env(width=3.37,height=3.37/ golden_ratio,\
                   lrbt=[0.18,0.88,0.25,0.95],fsize=9.0)
    for fk in fkeys:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        colors = plt.cm.brg(np.linspace(0,1,len(fkeys[fk])))
        lcolors = {fkeys[fk][i]:colors[i] for i in range(len(fkeys[fk]))}
        
        for sk in fkeys[fk]:
            d = dsol[sk]
            ax.plot(d[1:,0]/1000., d[1:,1], ls='--', lw=0.5, color=lcolors[sk], label=r'%s'%('-'.join(sk.split('_')[2:])))

        ax.set_xlabel(r'simulation time (ps)')
        ax.set_ylabel(r'energy / eV')
        ax.legend(loc=2,prop={'size':5})
        writefig(filename+'_'+fk)
        
def get_av_msd_water(atoms,lag=0.5):
    #tool = mdtools()
    water = identify_water(atoms[0])
    msd_dict = tool._calc_msd_time_averaged(xlayer,atypes=[3,8,17],lag=0.5)
    #TODO: hier weiter

def _get_diffusion_coefficient(msd):
    #TODO: to adjust
    for el in [3,8,17]:
        ta_av = ta[:msd_av[el][:,0].size]
        alpha_av = self._fit_rates(np.log(ta_av[2:]-ta_av[1]),np.log(msd_av[el][2:,3]-msd_av[el][1,3])) #start-ind=1 exclude intial offset-vibration
        dfactor_av = (msd_av[el][-1,3]-msd_av[el][1,3])/(2*3*(ta_av[-1]-ta_av[1])) #start-ind=1 exclude intial offset-vibration
        mdat_av.append([1000*dfactor_av,alpha_av])

        msd_lin, poplin = self._get_linear_behaviour(ta_av,msd_av[el][:,3]) #get where linear
        #TODO: is 2*6 correct? not only 6 = 2*3
        dfactor_av_lin = (msd_lin[1][-1]-msd_lin[1][1])/(2*3*(msd_lin[0][-1]-msd_lin[0][1])) #start-ind=1 exclude intial offset-vibration
        alpha_av_lin = self._fit_rates(np.log(msd_lin[0][2:]-msd_lin[0][1]),np.log(msd_lin[1][2:]-msd_lin[1][1])) #start-ind=1 exclude intial offset-vibration
        mdat_av_lin.append([1000*dfactor_av_lin,alpha_av_lin])

def _plot_msd_dict(self,filename,msd_dict,t,vert,target_folder='output'):
    #TODO: adjust
    f_lin = lambda x, a, b: a + b*x
    #NOTE: msd_dict dictionary of el with corresponding msd
    colors=[tumcolors["tumblue"],tumcolors["diag_red"],tumcolors["tumgreen"]]
    el = np.sort(msd_dict.keys())
    #self._set_plotting_env(3.37,3.37/ golden_ratio *1.5/2,[0.18,0.95,0.25,0.95])
    self._set_plotting_env(2.8,3.37/ golden_ratio *1.5/2.,[0.22,0.95,0.30,0.95],fsize=10.,font='lmodern')#'libertine')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    mmax = 0
    for i in range(0,len(el)):
        #msd_av = self.tool._calc_msd_time_averaged(atoms,el[i],lag=0.5)
        msd = msd_dict[el[i]]
        mmax = max(mmax,msd[:,3].max())
        ax1.plot(t[::4],msd[::4,3],color=colors[i],label=r'%s'%chemical_symbols[el[i]])
        ax1.plot([t[0],t[-1]],[msd[1,3],msd[1,3]],color=colors[i],ls=':',lw=1) #indicating the intial offset-vibration
        msd_lin, pop_lin = self._get_linear_behaviour(t,msd[:,3])
        ax1.plot(msd_lin[0],f_lin(msd_lin[0],pop_lin[0],pop_lin[1]),ls='--',color=tumcolors['tumorange'])
    if (vert != None):
        ax1.axvline(x=vert,ymin=0,ymax=mmax,color='k')
   #ax1.legend(loc=2,prop={'size':5})
   #ax1.legend(loc=2,prop={'size':9})
   #majorLocator = MultipleLocator(400)
 ###ax1.yaxis.set_major_locator(MultipleLocator(0.5))
 ###ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
 ###ax1.set_ylim(0.0,2.0)
 ###ax1.set_xlim(0.0,1700.)
    ax1.set_xlabel(r'lag time / ps')
    ax1.set_ylabel(r'msd / \AA $^2$')
    matplotlibhelpers.write(filename,folder=target_folder,write_info = False,write_png=False,write_pdf=True,write_eps=False)


if __name__ == "__main__":

    folder = sys.argv[1]

    dat = get_OUTCAR_energetics(folder+"/OUTCAR")
    plot_engs_vs_time("EvsT_%s"%(folder),dat)
    plot_av_engs_vs_time("EvsT_av_%s"%(folder),dat)
    if not os.path.isfile("output/density_%s.pdf"%folder):
        ions = []
        if folder.find("Na+") != -1:
            ions = [11]
        binc, hist_dicts = _get_density(folder,ions)
        plot_density("density_%s"%folder,binc,hist_dicts)




