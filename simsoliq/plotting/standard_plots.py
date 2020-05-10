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


def set_plotting_env(width=3.37,height=3.37/ golden_ratio *1.5/2,\
                    lrbt=[0.135,0.80,0.25,0.95],fsize=9.0):
    """
      function to set some default parameters for formatting of standard plots

      Parameters
      ----------
      width : float
          width of the plot in inches
      height : float
          height of the plot in inches
      lrbt : list/tuple with 4 entries
          left, right, bottom, top (fractional) margin of plot
      fsize : float
          font size in the plot
 
    """
    # set plot geometry
    rcParams['figure.figsize'] = (width, height) # x,y
    rcParams['font.size'] = fsize
    rcParams['figure.subplot.left'] = lrbt[0]  # the left side ..
    rcParams['figure.subplot.right'] = lrbt[1] # the right side ...
    rcParams['figure.subplot.bottom'] = lrbt[2] # the bottom ...
    rcParams['figure.subplot.top'] = lrbt[3] # the top ...
                                            # ... of the subplots of the figure
    rcParams['xtick.top'] = True
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.right'] = True
    rcParams['ytick.direction'] = 'in'

    rcParams['legend.fancybox'] = False
    rcParams['legend.edgecolor'] = 'k'


def writefig(filename, folder='output',  write_eps=False):
    """
      wrapper for creating figures
      
      Parameters
      ----------
      filename : string
        name of the produced figure (without extention)
      folder : string 
        subfolder in which to write the figure (default = output)
      write_eps : bool
        whether to create an eps figure (+ the usual pdf)

    """
    # folder for output
    if not os.path.isdir(folder):
        os.makedirs(folder)

    fileloc = os.path.join(folder, filename)
    print("writing {}".format(fileloc+'.pdf'))
    plt.savefig(fileloc+'.pdf')
    
    if write_eps:
        print("writing {}".format(fileloc+'.eps'))
        plt.savefig(fileloc+'.eps')
    
    plt.close(plt.gcf())


#############################################################################
############## plotting functions for energy vs. time plots #################
#############################################################################

# global parameters for time conversion
tdict = {'ns':1e-9, 'ps':1e-12, 'fs':1e-15}


def plot_engs_vs_time(filename, dat, tstart, tunit='ps'):
    """
      funtion to plot energy vs. time for individual trajectories 
      
      Parameters
      ----------
      filename : string
        name of the produced figure (without extention)
      dat : dict
        dictionary with energy data as given by an mdtraj object
      tstart : int
        cutoff at which data point to start plotting
      tunit : str
        unit of timestep for plotting (fs, ps, ...)

    """
    label = {'ekin':r'E$_{\mathrm{kin}}$ (eV)',\
            'epot':r'E$_{\mathrm{pot}}$ (eV)',\
            'etot':r'E$_{\mathrm{tot}}$ (eV)',\
            'x':r'simulation time (%s)'%tunit}
    _plot_xe_vs_time(filename,dat,label,_nofunc,tstart=tstart,tunit=tunit)


def _nofunc(indat):
    return(indat)


def plot_av_engs_vs_time(filename, dat, tstart, tunit='ps'):
    """
      funtion to plot running average of energy vs. time for 
      individual trajectories 
      
      Parameters
      ----------
      filename : string
        name of the produced figure (without extention)
      dat : dict
        dictionary with energy data as given by an mdtraj object
      tstart : int
        cutoff at which data point to start plotting
      tunit : str
        unit of timestep for plotting (fs, ps, ...)

    """
    label = {'ekin':r'$\langle$E$_{\mathrm{kin}} \rangle$ (eV)',\
            'epot':r'$\langle$E$_{\mathrm{pot}} \rangle$ (eV)',\
            'etot':r'E$_{\mathrm{tot}}$ (eV)',\
            'x':r'simulation time (%s)'%tunit}
    _plot_xe_vs_time(filename,dat,label,_running_av,tstart=tstart,tunit=tunit)


def _running_av(indat):
    y = np.cumsum(indat)/np.arange(1,indat.size+1)
    return(y)


def _plot_xe_vs_time(filename,dat,label,yfunc,tstart=1e3,tunit='ps'):
    """ 
      basic plotting function of property vs. time 
    
    """
    tfactor = tdict[dat['timeunit']] / tdict[tunit]
    ekeys = [key for key in dat if key[0] == 'e']

    set_plotting_env(width=3.37,height=3.37*(len(ekeys)/6.+0.5),
                   lrbt=[0.25,0.95,0.15,0.95],fsize=9.0)
    
    fig = plt.figure()
    n = 0; axes = []
    for e in ekeys:
        ax = fig.add_subplot(len(ekeys)*100+11+n)
        t = np.arange(tstart,dat[e].size)*tfactor
        y = yfunc(dat[e][int(tstart):])
        pmult = max(1,int(dat[e].size/1000.))
        ax.plot(t[::pmult],y[::pmult],color='k')
        ax.set_ylabel(label[e])
        axes.append(ax)
        n += 1
        # add line indicating AIMD traj restarts
        if 'runlengths' in dat:
            ttj = 0.0
            for tt in dat['runlengths']:
                ttj += tt
                ax.axvline(x=ttj*tfactor,lw=0.5,ls=':',color='k')
    axes[-1].set_xlabel(label['x'])
    [axes[a].set_xticklabels([]) for a in range(len(axes)-1)]
    plt.subplots_adjust(hspace=0.0)
    writefig(filename)


#############################################################################
#############################################################################
#############################################################################


#############################################################################
################# plotting functions for density profile  ###################
#############################################################################


def plot_density(filename, binc, hist_dicts, integral={}, dens=False):
    """
      funtion to plot the density profile for individual or
      averaged trajectories 
      
      Parameters
      ----------
      filename : string
        name of the produced figure (without extention)
      binc : list/array
        bin centers of density histogram (height)
      hist_dicts : dict
        dictionary including each elements density histrogram
      integral : dict
        values to plot shaded region for integral
      dens : bool
        option to plot the density (water)

    """
    set_plotting_env(width=3.37,height=3.37/ golden_ratio *1.5/2,\
                   lrbt=[0.18,0.88,0.25,0.95],fsize=9.0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax, ax2 = _plot_density(binc, hist_dicts, ax, integral, dens)
    if ax2 != None:
        ax2.set_ylabel(r'density water (g/cm$^3$)')
    
    ax.set_xlabel(r'z ($\mathrm{\AA}$)')
    ax.set_ylabel(r'density (mol/cm$^3$)')
    ax.legend(loc='best',prop={'size':7})
    ax.set_xlim(0.0,ax.get_xlim()[1])
    
    writefig(filename)


def _plot_density(binc, hist_dicts, ax, integral={}, dens=False, \
    labels={'Osolv':'Ow', 'Hsolv':'Hw'}):
    """
      function to populate axis object, density is hardcoded
      to water would need to make general via n_el/sum(m_solv)

    """
    labels.update({key:key for key in hist_dicts if key not in labels})
    colors = {'Osolv':'r','Hsolv':'c'}
    colors.update({el:jmol_colors[symbols.index(el)] \
            for el in hist_dicts if el not in colors})
    for el in hist_dicts:
        nz = np.where(hist_dicts[el] != 0.0)[0]
        if nz.size != 0:
            ax.plot(binc[nz], hist_dicts[el][nz], ls='-',\
                color=colors[el], label=r'%s'%labels[el])
        if el in integral:
            indi = np.where(binc >= integral[el][0])[0][0]
            ax.fill_between(binc[nz[0]:indi], hist_dicts[el][nz[0]:indi], color=colors[el], alpha=0.3)
            xy = [binc[indi], hist_dicts[el][indi]]; xytext = [xy[0]+2.0, xy[1] + 0.05]
            ax.annotate(r'%.1e'%integral[el][1], xy=xy, xytext=xytext, size=8, color=colors[el],
                arrowprops=dict(arrowstyle="-",connectionstyle="arc",color=colors[el]))
    hlim = np.array(ax.get_xlim()); d = hlim[1]-hlim[0]; hlim[0] += 0.2*d; hlim[1] -= 0.2*d
    
    if dens: # probably to be taken out
        # water density
        ax2 = ax.twinx()
        ax2.hlines([1],*hlim,color='k',linestyle='--',linewidth=0.5)
        nz = np.where(hist_dicts['Ow'] != 0.0)[0]
        ax2.plot(binc[nz]-binc[nz][0],hist_dicts['Ow'][nz]*18,ls=':',color='k')
        ax.plot(np.nan, np.nan, ls=':', c='k', label=r'$\rho_{\mathrm{H_2O}}$')
    else:
        ax2 = None
        ax.hlines([1/18.],*hlim, color=colors['Osolv'], linestyle='--', linewidth=0.5)
        ax.hlines([2/18.],*hlim, color=colors['Hsolv'], linestyle='--', linewidth=0.5)
    return(ax, ax2)
    
#############################################################################
#############################################################################
#############################################################################


#############################################################################
################# miscellaneous plotting functions        ###################
#############################################################################

def plot_profile(filename, x, y, vertline=0.0, xmarkings=[], xlim=None, \
    ylabel='', xlabel=''):
    """
      funtion to plot the any generic profile i.e. electrostatic potential
      
      Parameters
      ----------
      filename : string
        name of the produced figure (without extention)
      x : list/array
        x values for profile
      y : list/array
        y values for profile
      vertline : float, optional
        where to draw a vertical line and define a new `zero`
      xlim : tuple, optional
        limit for x-values
      ylabel/xlabel : str
        axis labels

    """
    set_plotting_env(width=3.37,height=3.37/ golden_ratio,
                   lrbt=[0.2,0.95,0.2,0.95],fsize=9.0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plot curve
    ax.plot(x-vertline, y, color='k')

    # axis labels
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.axvline(0.0, ls='--', color='k')
    for x in xmarkings:
        ax.axvline(x, ls=':', color='k')
    if xlim != None:
        ax.set_xlim(xlim)

    writefig(filename)

