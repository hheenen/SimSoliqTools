"""
This module contains some helper functions to produce 
standard plots for MD-analysis at solid/liquid interfaces

"""

import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.constants import golden_ratio, inch

from ase.data.colors import jmol_colors
from ase.data import chemical_symbols as symbols

from simsoliq.plotting.standard_plots import set_plotting_env, writefig



# global parameters for time conversion
tdict = {'ns':1e-9, 'ps':1e-12, 'fs':1e-15}

def plot_running_av_ensemble(filename, edat, surftag = {}, cdict = {}, \
    tstart=5000, tunit='ps', timestep=1, timeunit='fs', folder='output'): 
    """
      function to produce a collective plot for running average of many 
      trajectories of different types
 
      Parameters
      ----------
      filename : str
        name of figure to produce
      edat : dict
        energy data belonging to one type of energy (`ekin`, `etot`, `epot`)
        as given by `sort_energies`
      surftag : dict
        possible annotations for plot
      tstart : float
        time from when to start computing running average
      tunit : str
        unit of time used for the plot
      timestep : float
        timestep of given md-data
      timeunit : str
        unit of time of given md-data
 
    """

    # rename tags to include `clean` tag
    keys = list(edat.keys())
    for key in keys:
        if len(key.split('_')) == 2:
            edat[key+'_clean'] = edat.pop(key)
    
    # make colors based on subtags
    subtags = list(set([key.split('_')[2] for key in edat]))
    clrlist = plt.rcParams['axes.prop_cycle'].by_key()['color']
    clr = {subtags[i]:clrlist[i] for i in range(len(subtags))}
    
    # sort for subplots and make tags if none are given (remove underscores)
    if len(cdict) == 0:
        clist = list(set(['_'.join(ck.split('_')[:-1]) for ck in edat])); clist.sort()
        cdict = {ck:{ekey for ekey in edat if ekey.find(ck) != -1} for ck in clist}
    else:
        clist = list(cdict.keys()); clist.sort()
    if len(surftag) == 0:
        surftag = {ck:'-'.join(ck.split('_')) for ck in clist}
    
    # plot layout
    set_plotting_env(width=3.37,height=3.37/golden_ratio,\
                   lrbt=[0.2,0.95,0.2,0.95],fsize=9.0)

    fig = plt.figure()
    tfactor = tdict[timeunit] / tdict[tunit]
    axes = []
    
    for c in range(len(clist)):
        ax = fig.add_subplot(101+10*len(cdict)+c)
        ax.annotate(r'%s'%surftag[clist[c]], xy=(0.15,0.65), xycoords='axes fraction',\
            size=9)
        for ekey in cdict[clist[c]]:
            for i in range(len(edat[ekey])):
                t = np.arange(0,edat[ekey][i].size*timestep,timestep)
                t = np.around(t*tfactor,3) # may become problematic
                istart = np.where(t == tstart)[0][0]
                # make running average
                y = np.cumsum(edat[ekey][i][istart:])/\
                    np.arange(1,edat[ekey][i][istart:].size+1)
                ax.plot(t[istart:], y, ls='-', color=clr[ekey.split('_')[2]])
        axes.append(ax)

    # legend
    for ads in clr:
        axes[0].plot(np.nan,np.nan, ls='-', color=clr[ads], label=r'%s'%ads)
    axes[0].legend(loc='best',prop={'size':6},bbox_to_anchor=(0.0,0.2))
    
    # axis labels
    axes[0].set_ylabel(r'$\langle \Delta E \rangle (t)$ (eV)')
    mdl = int((len(clist)/2)-1)
    axes[mdl].set_xlabel(r'time (%s)'%tunit)
    if len(axes)%2 == 0:
        axes[mdl].xaxis.set_label_coords(1.0, -0.18)

   ## shared yaxis
   #ylim = [min([ax.get_ylim()[0] for ax in axes]),\
   #    max([ax.get_ylim()[1] for ax in axes])]
   #[ax.set_ylim(ylim) for ax in axes]
   #[axes[a].set_yticklabels([]) for a in range(1,len(axes))]
   #plt.subplots_adjust(wspace=0.05)

    writefig(filename, folder=folder)



